from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import ollama
import os
import logging
import torch
import clip

logger = logging.getLogger(__name__)

class TacticalAnalyzer:
    def __init__(self, llm, yolo_model, clip_model, clip_preprocess, device):
        """Initialize with YOLO, CLIP, and LLaVA"""
        self.llm = llm
        self.yolo_model = yolo_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        # Load YOLO
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            logger.info("✓ YOLO loaded")
        except Exception as e:
            logger.error(f"YOLO failed: {e}")
            self.yolo_model = None
        
        # Load CLIP
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info(f"✓ CLIP loaded on {self.device}")
        except Exception as e:
            logger.error(f"CLIP failed: {e}")
            self.clip_model = None
        
        # Tactical classifications for CLIP
        self.terrain_types = [
            "dense forest with trees",
            "open grassland field",
            "mountainous rocky terrain",
            "urban city buildings",
            "desert sandy area",
            "river or water body",
            "snowy winter landscape",
            "agricultural farmland",
            "hilly rolling terrain",
            "coastal beach area"
        ]
        
        self.tactical_features = [
            "high ground elevation",
            "defensive position with cover",
            "open killzone with no cover",
            "narrow chokepoint passage",
            "wide open flanking route",
            "dense vegetation concealment",
            "elevated observation post",
            "natural barrier obstacle"
        ]
        
        self.object_categories = [
            "military vehicles",
            "buildings and structures",
            "roads and pathways",
            "natural obstacles",
            "fortifications",
            "open areas"
        ]
    
    def classify_with_clip(self, image_path, text_options, top_k=3):
        """Classify image against multiple text options using CLIP"""
        if self.clip_model is None:
            return []
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode text options
            text_tokens = clip.tokenize(text_options).to(self.device)
            
            # Get similarity scores
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get top k results
            values, indices = similarity[0].topk(top_k)
            
            results = []
            for i, (value, idx) in enumerate(zip(values, indices)):
                results.append({
                    'label': text_options[idx],
                    'confidence': float(value),
                    'rank': i + 1
                })
            
            return results
            
        except Exception as e:
            logger.error(f"CLIP classification failed: {e}")
            return []
    
    def analyze_terrain_with_clip(self, image_path):
        """Use CLIP to identify terrain types"""
        logger.info("Analyzing terrain with CLIP...")
        
        terrain_results = self.classify_with_clip(image_path, self.terrain_types, top_k=3)
        tactical_results = self.classify_with_clip(image_path, self.tactical_features, top_k=3)
        object_results = self.classify_with_clip(image_path, self.object_categories, top_k=2)
        
        analysis = {
            'terrain_type': terrain_results,
            'tactical_features': tactical_results,
            'objects_present': object_results
        }
        
        # Log results
        logger.info("CLIP Terrain Analysis:")
        logger.info(f"  Primary terrain: {terrain_results[0]['label']} ({terrain_results[0]['confidence']:.2%})")
        logger.info(f"  Key feature: {tactical_results[0]['label']} ({tactical_results[0]['confidence']:.2%})")
        
        return analysis
    
    def analyze_regions_with_clip(self, image_path):
        """Analyze different regions of the map with CLIP"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # Divide into 3x3 grid
            regions = {}
            directions = [
                ['northwest', 'north', 'northeast'],
                ['west', 'center', 'east'],
                ['southwest', 'south', 'southeast']
            ]
            
            for i in range(3):
                for j in range(3):
                    # Extract region
                    x1 = j * (width // 3)
                    y1 = i * (height // 3)
                    x2 = (j + 1) * (width // 3)
                    y2 = (i + 1) * (height // 3)
                    
                    region_img = img.crop((x1, y1, x2, y2))
                    
                    # Save temporarily
                    temp_path = f"/tmp/region_{i}_{j}.jpg"
                    region_img.save(temp_path)
                    
                    # Classify region
                    terrain = self.classify_with_clip(temp_path, self.terrain_types, top_k=1)
                    tactical = self.classify_with_clip(temp_path, self.tactical_features, top_k=1)
                    
                    if terrain and tactical:
                        regions[directions[i][j]] = {
                            'terrain': terrain[0]['label'],
                            'terrain_confidence': terrain[0]['confidence'],
                            'tactical_feature': tactical[0]['label'],
                            'tactical_confidence': tactical[0]['confidence']
                        }
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            return regions
            
        except Exception as e:
            logger.error(f"Region analysis failed: {e}")
            return {}
    
    def detect_objects_yolo(self, image_path):
        """YOLO object detection (existing code)"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(image_path, conf=0.25)
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    class_name = result.names[cls]
                    
                    detections.append({
                        'type': class_name,
                        'confidence': round(conf, 2),
                        'position': (center_x, center_y),
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
            
            logger.info(f"YOLO detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def get_visual_understanding_llava(self, image_path, scenario):
        """LLaVA visual analysis (existing code)"""
        try:
            logger.info("Querying LLaVA for detailed visual analysis...")
            
            response = ollama.chat(
                model='llava:13b',
                messages=[{
                    'role': 'user',
                    'content': f"""Analyze this tactical/battlefield map in detail.

Scenario: {scenario}

Provide a comprehensive description of:
1. Terrain features and layout
2. Strategic positions (high ground, cover, chokepoints)
3. Natural and man-made obstacles
4. Potential approach routes
5. Defensive and offensive positions
6. Any visible units or structures

Be specific about locations using compass directions, altitude changes, and terrain types.
Always ask the user to confirm if the information is correct or if they have additional details to provide.""",
                    'images': [image_path]
                }]
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"LLaVA failed: {e}")
            return "Visual analysis unavailable."
    
    def generate_comprehensive_strategy(self, image_path, scenario, unit_types=None):
        """Complete tactical analysis using all three models"""
        
        logger.info("="*70)
        logger.info("MULTI-MODEL TACTICAL ANALYSIS")
        logger.info("="*70)
        
        if unit_types is None:
            unit_types = {
                'infantry': 20,
                'tanks': 5,
                'artillery': 3,
                'reconnaissance': 2
            }
        
        # PHASE 1: CLIP - Fast terrain classification
        logger.info("\n[1/4] CLIP: Fast Terrain Classification")
        clip_terrain = self.analyze_terrain_with_clip(image_path)
        clip_regions = self.analyze_regions_with_clip(image_path)
        
        # PHASE 2: YOLO - Object detection
        logger.info("\n[2/4] YOLO: Object Detection")
        yolo_objects = self.detect_objects_yolo(image_path)
        
        # PHASE 3: LLaVA - Detailed visual understanding
        logger.info("\n[3/4] LLaVA: Detailed Visual Analysis")
        llava_analysis = self.get_visual_understanding_llava(image_path, scenario)
        
        # PHASE 4: Synthesize with Llama
        logger.info("\n[4/4] Llama 3.2: Strategy Synthesis")
        
        # Build comprehensive intelligence report
        clip_summary = f"""CLIP TERRAIN ANALYSIS:
Primary Terrain: {clip_terrain['terrain_type'][0]['label']} ({clip_terrain['terrain_type'][0]['confidence']:.1%} confidence)
Secondary: {clip_terrain['terrain_type'][1]['label']} ({clip_terrain['terrain_type'][1]['confidence']:.1%})

Key Tactical Features:
1. {clip_terrain['tactical_features'][0]['label']} ({clip_terrain['tactical_features'][0]['confidence']:.1%})
2. {clip_terrain['tactical_features'][1]['label']} ({clip_terrain['tactical_features'][1]['confidence']:.1%})

Regional Breakdown:"""
        
        for direction, info in clip_regions.items():
            clip_summary += f"\n  {direction.upper()}: {info['terrain']} - {info['tactical_feature']}"
        
        yolo_summary = f"\nYOLO OBJECT DETECTION:\n"
        if yolo_objects:
            yolo_summary += f"Detected {len(yolo_objects)} objects:\n"
            for obj in yolo_objects[:10]:  # Limit to 10
                yolo_summary += f"  - {obj['type']} at ({obj['position'][0]}, {obj['position'][1]}) [confidence: {obj['confidence']}]\n"
        else:
            yolo_summary += "No objects detected\n"
        
        # Generate strategy
        strategy_prompt = f"""You are a smart military strategist with AI-enhanced battlefield intelligence.

SCENARIO: {scenario}

AVAILABLE FORCES:
{chr(10).join([f"- {unit}: {count} units" for unit, count in unit_types.items()])}

=== MULTI-SOURCE INTELLIGENCE REPORT ===

{clip_summary}

{yolo_summary}

LLAVA DETAILED RECONNAISSANCE:
{llava_analysis}

=== END REPORT ===

Using the three-source intelligence above (CLIP terrain classification, YOLO object detection, and LLaVA visual analysis), create a detailed tactical deployment plan.

Your strategy should:

1. TERRAIN EXPLOITATION
   - Leverage CLIP's terrain identification for unit placement
   - Match unit types to terrain advantages
   - Identify cover, concealment, and fields of fire

2. THREAT ASSESSMENT
   - Account for detected objects from YOLO
   - Plan around existing structures/obstacles
   - Identify potential enemy positions

3. FORCE DEPLOYMENT
   For each unit type, specify:
   - Exact positions using compass directions and terrain features
   - Tactical rationale based on intelligence
   - Coordination with other units
   - Fallback positions

4. EXECUTION PLAN
   - Priority of deployment
   - Critical terrain to secure first
   - Coordination sequence
   - Contingencies

Be highly specific. Reference the intelligence findings directly.
In order for you to provide the best possible strategy, ensure you integrate insights from all three models cohesively.
If the models do not recognise any information on the map, then let the user know that no viable strategy can be formed based on the available data.
If only partial information or even in the cases of no information ask the user questions to understand the situation better and be
able to create a more informed strategy. If the user does not have enough information then let them know that no viable strategy can be formed based on the available data.
If the user does not provide any information about their units then inquire about their available forces before proceeding.
If you do not understand something then ask the user for clarification before proceeding always."""

        strategy = self.llm.invoke(strategy_prompt)
        
        logger.info("="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70)
        
        return {
            'strategy': strategy,
            'clip_analysis': clip_terrain,
            'clip_regions': clip_regions,
            'yolo_detections': yolo_objects,
            'llava_analysis': llava_analysis,
            'models_used': ['CLIP', 'YOLO', 'LLaVA', 'Llama 3.2']
        }
    
    def create_annotated_map(self, image_path, yolo_detections, clip_regions, output_path):
        """Create map with YOLO detections and CLIP region labels"""
        try:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            
            # Draw YOLO detections
            for detection in yolo_detections:
                x1, y1, x2, y2 = detection['bbox']
                label = f"{detection['type']} ({detection['confidence']})"
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw CLIP region labels
            directions = [
                ['northwest', 'north', 'northeast'],
                ['west', 'center', 'east'],
                ['southwest', 'south', 'southeast']
            ]
            
            for i in range(3):
                for j in range(3):
                    direction = directions[i][j]
                    if direction in clip_regions:
                        region_info = clip_regions[direction]
                        
                        # Calculate region center
                        x = int((j + 0.5) * (width / 3))
                        y = int((i + 0.5) * (height / 3))
                        
                        # Draw label
                        terrain = region_info['terrain'].split()[0]  # First word
                        label = f"{direction.upper()[:2]}: {terrain}"
                        
                        cv2.putText(img, label, (x - 40, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imwrite(output_path, img)
            logger.info(f"Annotated map saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Map annotation failed: {e}")
            return None