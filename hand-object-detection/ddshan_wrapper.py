"""
Wrapper para integrar el detector de ddshan/hand_object_detector
https://github.com/ddshan/hand_object_detector

Este wrapper permite usar el detector de Faster R-CNN como alternativa
al método actual basado en MediaPipe + YOLO.
"""

import os
import sys
import cv2
import numpy as np
import torch

# Intentar importar el detector de ddshan
DDSHAN_AVAILABLE = False
try:
    # Añadir el path del detector de ddshan
    ddshan_path = os.path.join(os.path.dirname(__file__), 'ddshan_detector')
    if os.path.exists(ddshan_path):
        sys.path.insert(0, ddshan_path)
        sys.path.insert(0, os.path.join(ddshan_path, 'lib'))
        
        # Intentar importar módulos necesarios
        import _init_paths
        from model.utils.config import cfg, cfg_from_file
        from model.faster_rcnn.resnet import resnet
        from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
        from model.roi_layers import nms
        from model.utils.blob import im_list_to_blob
        from model.utils.net_utils import vis_detections_filtered_objects
        
        DDSHAN_AVAILABLE = True
        print("✓ Detector de ddshan disponible")
except Exception as e:
    print(f"⚠ Detector de ddshan no disponible: {e}")
    print("  Usando método alternativo (MediaPipe + YOLO)")


class DdshanDetector:
    """
    Wrapper para el detector de ddshan/hand_object_detector
    """
    
    def __init__(self, model_path=None, checkpoint=132028, checkepoch=1, checksession=1, 
                 net='res101', thresh_hand=0.5, thresh_obj=0.5, cuda=True):
        """
        Inicializa el detector de ddshan
        
        Args:
            model_path: Ruta al modelo pre-entrenado (si None, intenta encontrarlo)
            checkpoint: Número de checkpoint del modelo
            checkepoch: Época del checkpoint
            checksession: Sesión del checkpoint
            net: Red a usar ('res101', 'res50', 'vgg16')
            thresh_hand: Umbral de confianza para manos
            thresh_obj: Umbral de confianza para objetos
            cuda: Usar CUDA si está disponible
        """
        if not DDSHAN_AVAILABLE:
            raise ImportError("El detector de ddshan no está disponible. "
                            "Asegúrate de tener compilado el código CUDA y los modelos descargados.")
        
        self.thresh_hand = thresh_hand
        self.thresh_obj = thresh_obj
        self.cuda = cuda and torch.cuda.is_available()
        
        # Configurar paths
        ddshan_path = os.path.join(os.path.dirname(__file__), 'ddshan_detector')
        cfg_file = os.path.join(ddshan_path, 'cfgs', f'{net}.yml')
        
        if os.path.exists(cfg_file):
            cfg_from_file(cfg_file)
        else:
            # Configuración por defecto
            cfg_file = os.path.join(ddshan_path, 'cfgs', 'res101.yml')
            if os.path.exists(cfg_file):
                cfg_from_file(cfg_file)
        
        cfg.USE_GPU_NMS = self.cuda
        
        # Configurar clases
        self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
        cfg.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']
        
        # Inicializar modelo
        if net == 'res101':
            self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=False)
        elif net == 'res50':
            self.fasterRCNN = resnet(self.pascal_classes, 50, pretrained=False, class_agnostic=False)
        else:
            raise ValueError(f"Red {net} no soportada. Usa 'res101' o 'res50'")
        
        self.fasterRCNN.create_architecture()
        
        # Cargar modelo
        if model_path is None:
            # Intentar encontrar el modelo en la ubicación estándar
            model_dir = os.path.join(ddshan_path, 'models', f'{net}_handobj_100K', 'pascal_voc')
            model_path = os.path.join(model_dir, 
                                     f'faster_rcnn_{checksession}_{checkepoch}_{checkpoint}.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo no encontrado en: {model_path}\n"
                f"Por favor, descarga el modelo desde:\n"
                f"https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE\n"
                f"Y guárdalo en: {model_path}"
            )
        
        print(f"Cargando modelo desde: {model_path}")
        if self.cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        
        print('✓ Modelo cargado correctamente')
        
        # Preparar tensores
        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)
        self.box_info = torch.FloatTensor(1)
        
        if self.cuda:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()
            cfg.CUDA = True
            self.fasterRCNN.cuda()
        
        self.fasterRCNN.eval()
    
    def _get_image_blob(self, im):
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
    
    def detect(self, frame):
        """
        Detecta manos y objetos en un frame
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            dict con:
                - 'hands': Lista de detecciones de manos [(x1, y1, x2, y2, conf, state, side), ...]
                - 'objects': Lista de detecciones de objetos [(x1, y1, x2, y2, conf), ...]
                - 'frame_annotated': Frame con visualizaciones
        """
        im = frame.copy()
        
        # Preprocesar imagen
        blobs, im_scales = self._get_image_blob(im)
        assert len(im_scales) == 1, "Solo se implementa batch de una imagen"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], 
                             dtype=np.float32)
        
        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)
        
        with torch.no_grad():
            self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            self.gt_boxes.resize_(1, 1, 5).zero_()
            self.num_boxes.resize_(1).zero_()
            self.box_info.resize_(1, 1, 5).zero_()
        
        # Detectar
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = self.fasterRCNN(
            self.im_data, self.im_info, self.gt_boxes, self.num_boxes, self.box_info
        )
        
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        
        # Extraer parámetros predichos
        contact_vector = loss_list[0][0]  # Estado de contacto de la mano
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
                if self.cuda:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))
            
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        
        pred_boxes /= im_scales[0]
        
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        
        # Procesar detecciones
        hand_dets = []
        obj_dets = []
        
        for j in range(1, len(self.pascal_classes)):
            if self.pascal_classes[j] == 'hand':
                inds = torch.nonzero(scores[:, j] > self.thresh_hand).view(-1)
                thresh = self.thresh_hand
            elif self.pascal_classes[j] == 'targetobject':
                inds = torch.nonzero(scores[:, j] > self.thresh_obj).view(-1)
                thresh = self.thresh_obj
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
                
                dets_np = cls_dets.cpu().numpy()
                
                if self.pascal_classes[j] == 'targetobject':
                    obj_dets = dets_np
                elif self.pascal_classes[j] == 'hand':
                    hand_dets = dets_np
        
        # Formatear resultados
        hands = []
        for det in hand_dets:
            x1, y1, x2, y2, conf, state, offset_x, offset_y, offset_mag, side = det[:10]
            hands.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'state': int(state),  # 0=N, 1=S, 2=O, 3=P, 4=F
                'side': 'left' if side < 0.5 else 'right',
                'offset': [float(offset_x), float(offset_y), float(offset_mag)]
            })
        
        objects = []
        for det in obj_dets:
            x1, y1, x2, y2, conf = det[:5]
            objects.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf)
            })
        
        # Visualizar
        frame_annotated = vis_detections_filtered_objects(
            im.copy(), obj_dets, hand_dets, self.thresh_hand, self.thresh_obj
        )
        
        return {
            'hands': hands,
            'objects': objects,
            'frame_annotated': frame_annotated
        }


def is_available():
    """Verifica si el detector de ddshan está disponible"""
    return DDSHAN_AVAILABLE
