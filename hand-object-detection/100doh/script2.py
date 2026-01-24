#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para procesar videos con el modelo 100DOH (local, sin Roboflow)
Basado en: https://github.com/nripstein/Thesis-100-DOH
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time

# Añadir paths necesarios
ddshan_path = os.path.join(os.path.dirname(__file__), '..', 'ddshan_detector')
if os.path.exists(ddshan_path):
    sys.path.insert(0, ddshan_path)
    sys.path.insert(0, os.path.join(ddshan_path, 'lib'))

try:
    import _init_paths
except Exception as e:
    print(f"⚠️  Advertencia al importar _init_paths: {e}")
    print("Continuando de todas formas...")

try:
    from model.utils.config import cfg, cfg_from_file
    from model.faster_rcnn.resnet import resnet
    from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
    from model.roi_layers import nms
    from model.utils.blob import im_list_to_blob
    from model.utils.net_utils import vis_detections_filtered_objects
    import torch
    from torch.autograd import Variable
    DDSHAN_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importando módulos de 100DOH: {e}")
    print("Asegúrate de tener compilado el código CUDA y las dependencias instaladas.")
    print("Ejecuta: cd ddshan_detector/lib && python setup.py build develop")
    sys.exit(1)


def _get_image_blob(im):
    """Convierte una imagen en entrada para la red"""
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                       interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def load_model(model_path, net='res101', checksession=1, checkepoch=1, checkpoint=132028, cuda=True):
    """Carga el modelo 100DOH"""
    print(f"🔄 Cargando modelo 100DOH ({net})...")
    
    # Configurar paths
    cfg_file = os.path.join(ddshan_path, 'cfgs', f'{net}.yml')
    if os.path.exists(cfg_file):
        cfg_from_file(cfg_file)
    else:
        print(f"⚠️  Archivo de configuración {cfg_file} no encontrado, usando valores por defecto")
    
    cfg.USE_GPU_NMS = cuda and torch.cuda.is_available()
    
    # Clases
    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    cfg.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']
    
    # Inicializar red
    if net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
    elif net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=False)
    else:
        raise ValueError(f"Red {net} no soportada. Usa 'res101' o 'res50'")
    
    fasterRCNN.create_architecture()
    
    # Cargar checkpoint
    if model_path is None:
        model_dir = os.path.join(ddshan_path, 'models', f'{net}_handobj_100K', 'pascal_voc')
        model_path = os.path.join(model_dir, f'faster_rcnn_{checksession}_{checkepoch}_{checkpoint}.pth')
        
        # Si no existe con los parámetros especificados, buscar cualquier modelo .pth en el directorio
        if not os.path.exists(model_path):
            import glob
            model_files = glob.glob(os.path.join(model_dir, 'faster_rcnn_*.pth'))
            if model_files:
                model_path = model_files[0]
                print(f"⚠️  Modelo encontrado con nombre diferente: {os.path.basename(model_path)}")
            else:
                raise FileNotFoundError(
                    f"❌ Modelo no encontrado en: {model_dir}\n"
                    f"Por favor, descarga el modelo desde:\n"
                    f"https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE\n"
                    f"Y guárdalo en: {model_dir}"
                )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Modelo no encontrado en: {model_path}\n"
            f"Por favor, descarga el modelo desde:\n"
            f"https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE\n"
            f"Y guárdalo en: {model_path}"
        )
    
    print(f"📦 Cargando checkpoint: {model_path}")
    use_cuda = cuda and torch.cuda.is_available()
    if use_cuda:
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    
    # Preparar tensores
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)
    
    if use_cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        cfg.CUDA = True
        fasterRCNN.cuda()
    
    fasterRCNN.eval()
    print("✓ Modelo cargado correctamente")
    
    return fasterRCNN, im_data, im_info, num_boxes, gt_boxes, box_info, pascal_classes, use_cuda


def process_frame(fasterRCNN, frame, im_data, im_info, num_boxes, gt_boxes, box_info, 
                  pascal_classes, thresh_hand=0.5, thresh_obj=0.5, use_cuda=True):
    """Procesa un frame y devuelve el frame anotado"""
    im = frame.copy()
    
    # Preprocesar imagen
    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Solo se implementa batch de una imagen"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    
    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)
    
    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        box_info.resize_(1, 1, 5).zero_()
    
    # Detectar
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)
    
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    
    # Extraer parámetros predichos
    contact_vector = loss_list[0][0]  # Estado de contacto
    offset_vector = loss_list[1][0].detach()  # Vector de offset
    lr_vector = loss_list[2][0].detach()  # Lado de la mano (izquierda/derecha)
    
    # Obtener estado de contacto
    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
    
    # Obtener lado de la mano
    lr = torch.sigmoid(lr_vector) > 0.5
    lr = lr.squeeze(0).float()
    
    # Aplicar regresión de bounding box
    if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            if use_cuda:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
        
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    
    pred_boxes /= im_scales[0]
    
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    
    # Procesar detecciones
    obj_dets, hand_dets = None, None
    for j in range(1, len(pascal_classes)):
        if pascal_classes[j] == 'hand':
            inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
        elif pascal_classes[j] == 'targetobject':
            inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)
        else:
            continue
        
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((
                cls_boxes, 
                cls_scores.unsqueeze(1), 
                contact_indices[inds], 
                offset_vector.squeeze(0)[inds], 
                lr[inds]
            ), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            
            if pascal_classes[j] == 'targetobject':
                obj_dets = cls_dets.cpu().numpy()
            elif pascal_classes[j] == 'hand':
                hand_dets = cls_dets.cpu().numpy()
    
    # Visualizar
    im2show = vis_detections_filtered_objects(im.copy(), obj_dets, hand_dets, thresh_hand, thresh_obj)
    
    return im2show


def main():
    parser = argparse.ArgumentParser(description='Procesar video con modelo 100DOH (local)')
    parser.add_argument('--video', type=str, required=True, help='Ruta al video de entrada')
    parser.add_argument('--model', type=str, default=None, help='Ruta al modelo (opcional)')
    parser.add_argument('--net', type=str, default='res101', choices=['res101', 'res50'], 
                       help='Red a usar (default: res101)')
    parser.add_argument('--checkpoint', type=int, default=132028, help='Número de checkpoint')
    parser.add_argument('--checkepoch', type=int, default=1, help='Época del checkpoint')
    parser.add_argument('--checksession', type=int, default=1, help='Sesión del checkpoint')
    parser.add_argument('--thresh_hand', type=float, default=0.5, help='Umbral para manos')
    parser.add_argument('--thresh_obj', type=float, default=0.5, help='Umbral para objetos')
    parser.add_argument('--cuda', action='store_true', help='Usar CUDA si está disponible')
    parser.add_argument('--no-display', action='store_true', help='No mostrar ventana de visualización')
    
    args = parser.parse_args()
    
    # Verificar video
    if not os.path.exists(args.video):
        print(f"❌ Error: No se encontró el archivo de video: {args.video}")
        sys.exit(1)
    
    # Cargar modelo
    try:
        fasterRCNN, im_data, im_info, num_boxes, gt_boxes, box_info, pascal_classes, use_cuda = \
            load_model(args.model, args.net, args.checksession, args.checkepoch, args.checkpoint, args.cuda)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        sys.exit(1)
    
    # Abrir video
    print(f"\n📹 Abriendo video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el archivo de video")
        sys.exit(1)
    
    # Obtener información del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}\n")
    
    # Crear archivo de salida
    video_dir = os.path.dirname(args.video) if os.path.dirname(args.video) else "."
    video_name = os.path.basename(args.video)
    video_name_no_ext, video_ext = os.path.splitext(video_name)
    output_video_path = os.path.join(video_dir, f"{video_name_no_ext}_100doh{video_ext}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print(f"💾 Guardando video procesado en: {output_video_path}\n")
    
    # Procesar frames
    frame_count = 0
    start_time = time.time()
    
    print("🔄 Procesando frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Procesar frame
        try:
            annotated_frame = process_frame(
                fasterRCNN, frame, im_data, im_info, num_boxes, gt_boxes, box_info,
                pascal_classes, args.thresh_hand, args.thresh_obj, use_cuda
            )
        except Exception as e:
            print(f"⚠️  Error procesando frame {frame_count}: {e}")
            annotated_frame = frame.copy()
        
        # Agregar información del frame
        frame_info = f"Frame: {frame_count}/{total_frames} | FPS: {fps}"
        cv2.putText(annotated_frame, frame_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Guardar frame
        out.write(annotated_frame)
        
        # Mostrar progreso
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            print(f"  Procesados: {frame_count}/{total_frames} frames ({fps_actual:.1f} fps)")
        
        # Mostrar ventana si no está deshabilitado
        if not args.no_display:
            cv2.imshow("100DOH - Detección de Manos y Objetos", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Procesamiento interrumpido por el usuario")
                break
    
    # Finalizar
    cap.release()
    out.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    
    elapsed_total = time.time() - start_time
    print(f"\n✅ Procesamiento completado!")
    print(f"   Frames procesados: {frame_count}/{total_frames}")
    print(f"   Tiempo total: {elapsed_total:.2f}s")
    print(f"   FPS promedio: {frame_count/elapsed_total:.2f}" if elapsed_total > 0 else "")
    print(f"   Video guardado en: {output_video_path}")


if __name__ == "__main__":
    main()
