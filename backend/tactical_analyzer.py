from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import ollama
import os
import logging
import torch
import clip
import tempfile
from symbol_helper import get_symbol_helper
from symbol_classifier import ResNetSymbolClassifier
import json

logger = logging.getLogger(__name__)

class TacticalAnalyzer:
    def __init__(self, llm, yolo_model, clip_model, clip_preprocess, device):
        """Initialize with YOLO, CLIP, and LLaVA"""
        self.llm = llm
        self.yolo_model = yolo_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        self.symbol_helper = get_symbol_helper()

        # Initialize NATO symbol classifier
        self.symbol_classifier = None
        self.class_mapping = None
        self._load_symbol_classifier()

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

    def _load_symbol_classifier(self):
        """Load trained symbol classifier if available"""
        try:
            model_path = './models/symbol_classifier.pth'
            mapping_path = './symbol_dataset/class_mapping.json'

            if os.path.exists(model_path) and os.path.exists(mapping_path):
                # Load class mapping first to determine number of classes
                with open(mapping_path, 'r') as f:
                    self.class_mapping = json.load(f)

                num_classes = len(self.class_mapping)
                self.symbol_classifier = ResNetSymbolClassifier(num_classes=num_classes, device=str(self.device))
                self.symbol_classifier.load_model(model_path)

                logger.info(f"NATO symbol classifier loaded successfully ({num_classes} classes)")
            else:
                logger.warning("Symbol classifier not found - symbol detection disabled")
                logger.warning(f"  Model: {model_path}")
                logger.warning(f"  Mapping: {mapping_path}")
        except Exception as e:
            logger.error(f"Failed to load symbol classifier: {e}")

    def detect_nato_symbols(self, image_path, yolo_detections):
        """Classify YOLO detections as NATO symbols"""
        if self.symbol_classifier is None or self.class_mapping is None:
            return []

        detected_symbols = []

        try:
            img = Image.open(image_path)

            for detection in yolo_detections:
                # Extract crop from detection bbox
                x1, y1, x2, y2 = detection['bbox']
                crop = img.crop((x1, y1, x2, y2))

                # Classify symbol
                result = self.symbol_classifier.predict(crop)
                class_idx = result['class_idx']
                confidence = result['confidence']

                # Get symbol name from mapping
                symbol_name = self.class_mapping.get(str(class_idx), f"Unknown ({class_idx})")

                # Format symbol name for readability
                readable_name = symbol_name.replace('_', ' ').title()

                # Calculate position in compass terms
                img_width, img_height = img.size
                center_x, center_y = detection['position']

                # Determine sector
                h_third = img_width / 3
                v_third = img_height / 3

                if center_y < v_third:
                    vertical = "North"
                elif center_y < 2 * v_third:
                    vertical = "Center"
                else:
                    vertical = "South"

                if center_x < h_third:
                    horizontal = "West"
                elif center_x < 2 * h_third:
                    horizontal = ""
                else:
                    horizontal = "East"

                sector = f"{vertical}{horizontal}".strip()
                if sector == "Center":
                    sector = "Center"

                detected_symbols.append({
                    'symbol': readable_name,
                    'confidence': confidence,
                    'sector': sector,
                    'position': (center_x, center_y),
                    'low_confidence': confidence < 0.85
                })

            logger.info(f"Detected {len(detected_symbols)} NATO symbols")
            return detected_symbols

        except Exception as e:
            logger.error(f"Symbol detection failed: {e}")
            return []

    def classify_with_clip(self, image_path, text_options, top_k=3):
        """Classify image against multiple text options using CLIP"""
        if self.clip_model is None:
            return []
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_tokens = clip.tokenize(text_options).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
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
        
        logger.info("CLIP Terrain Analysis:")
        logger.info(f"  Primary terrain: {terrain_results[0]['label']} ({terrain_results[0]['confidence']:.2%})")
        logger.info(f"  Key feature: {tactical_results[0]['label']} ({tactical_results[0]['confidence']:.2%})")
        
        return analysis
    
    def analyze_regions_with_clip(self, image_path):
        """Analyze different regions of the map with CLIP"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            regions = {}
            directions = [
                ['northwest', 'north', 'northeast'],
                ['west', 'center', 'east'],
                ['southwest', 'south', 'southeast']
            ]
            
            for i in range(3):
                for j in range(3):
                    x1 = j * (width // 3)
                    y1 = i * (height // 3)
                    x2 = (j + 1) * (width // 3)
                    y2 = (i + 1) * (height // 3)
                    
                    region_img = img.crop((x1, y1, x2, y2))
                    
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        temp_path = tmp.name
                        region_img.save(temp_path)
                    
                    terrain = self.classify_with_clip(temp_path, self.terrain_types, top_k=1)
                    tactical = self.classify_with_clip(temp_path, self.tactical_features, top_k=1)
                    
                    if terrain and tactical:
                        regions[directions[i][j]] = {
                            'terrain': terrain[0]['label'],
                            'terrain_confidence': terrain[0]['confidence'],
                            'tactical_feature': tactical[0]['label'],
                            'tactical_confidence': tactical[0]['confidence']
                        }
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            return regions
            
        except Exception as e:
            logger.error(f"Region analysis failed: {e}")
            return {}
    
    def detect_objects_yolo(self, image_path):
        """YOLO object detection"""
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
        """LLaVA visual analysis with NATO symbology reference"""
        try:
            logger.info("Querying LLaVA for detailed visual analysis...")

            # Get symbol reference guide
            symbol_guide = self.symbol_helper.get_symbol_reference_guide()

            response = ollama.chat(
                model='llava:13b',
                messages=[{
                    'role': 'user',
                    'content': f"""You are a NATO reconnaissance analyst conducting Intelligence Preparation of the Battlefield (IPB). You are trained in NATO APP-6(E) military symbology and can identify standard military map symbols.

MISSION: {scenario}

{symbol_guide}

When you see military symbols on the map, identify:
- Unit type (infantry, armor, artillery, reconnaissance, engineer, headquarters, etc.)
- Affiliation (friendly/enemy/neutral - indicated by frame shape and color)
- Echelon/size (symbols above the icon showing unit size)
- Location on the map (use compass directions and terrain features)
- Tactical significance of placement relative to terrain

Conduct a systematic battlefield reconnaissance following IPB methodology:

STEP 1 - MILITARY SYMBOL IDENTIFICATION:
- What military symbols are present on the map?
- What unit types, sizes, and affiliations do they represent?
- Where are friendly forces positioned? Where are enemy forces?
- Are there headquarters, support units, or special symbols marked?

STEP 2 - TERRAIN ANALYSIS (OAKOC Framework):
- Observation & Fields of Fire: What are the key observation points and fields of fire?
- Avenues of Approach: Identify mobility corridors for mounted/dismounted movement
- Key Terrain: What terrain features provide tactical advantage?
- Obstacles: Natural/man-made obstacles that channelize or restrict movement
- Cover & Concealment: Where can forces find protection from fires and observation?

STEP 3 - TACTICAL GEOMETRY & SYMBOL PLACEMENT ANALYSIS:
- High ground positions and their commanding fields of view
- Why are units positioned where they are relative to terrain?
- Do symbol placements align with sound tactical principles?
- Chokepoints that restrict maneuver
- Dead ground and defilade positions
- Flanking routes and engagement areas

STEP 4 - FORCE EMPLOYMENT ASSESSMENT:
- How are forces currently arrayed based on symbols?
- Suitable locations for defensive positions
- Viable offensive approach routes
- Artillery firing positions and observation posts
- Assembly areas and support-by-fire positions

Use compass directions (N, NE, E, SE, S, SW, W, NW), estimate distances, and describe elevations. Think like a military planner assessing this battlefield and the force dispositions shown by military symbols.""",
                    'images': [image_path]
                }]
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"LLaVA failed: {e}")
            return "Visual analysis unavailable."
    
    def generate_comprehensive_strategy(self, image_path, scenario, unit_types=None, vectorstore=None):
        """Complete tactical analysis using all models + KB"""
        
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
        
        # PHASE 1: CLIP
        logger.info("\n[1/5] CLIP: Terrain Classification")
        clip_terrain = self.analyze_terrain_with_clip(image_path)
        clip_regions = self.analyze_regions_with_clip(image_path)
        
        # PHASE 2: YOLO + Symbol Classification
        logger.info("\n[2/5] YOLO: Object Detection")
        yolo_objects = self.detect_objects_yolo(image_path)

        # PHASE 2.5: NATO Symbol Classification
        logger.info("\n[2.5/5] ResNet18: NATO Symbol Classification")
        nato_symbols = self.detect_nato_symbols(image_path, yolo_objects)

        # PHASE 3: LLaVA
        logger.info("\n[3/5] LLaVA: Visual Analysis")
        llava_analysis = self.get_visual_understanding_llava(image_path, scenario)
        
        # PHASE 4: KB Retrieval (Enhanced with unit-specific queries)
        logger.info("\n[4/5] Knowledge Base: Doctrine Retrieval")
        kb_context = ""
        if vectorstore is not None:
            # Extract unit categories from LLaVA analysis for targeted doctrine retrieval
            unit_categories = self.symbol_helper.extract_unit_categories(llava_analysis)

            if unit_categories:
                logger.info(f"Detected unit types: {', '.join(unit_categories)}")
                # Generate enhanced query based on identified units
                doctrine_query = self.symbol_helper.generate_enhanced_doctrine_query(scenario, unit_categories)
            else:
                # Fallback to general query if no specific units detected
                doctrine_query = f"""How does NATO doctrine address {scenario}?
                What are the tactical principles for terrain analysis, force deployment,
                offensive and defensive operations, intelligence preparation of the battlefield,
                and military decision-making processes?"""

            logger.info(f"Doctrine query: {doctrine_query[:150]}...")
            doctrine_docs = vectorstore.similarity_search(doctrine_query, k=8)
            
            if doctrine_docs:
                kb_context = "\n\n=== NATO DOCTRINE & TACTICAL REFERENCES ===\n"
                for idx, doc in enumerate(doctrine_docs, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    kb_context += f"\n[Reference {idx}: {source}]\n{doc.page_content}\n"
                kb_context += "\n=== END DOCTRINE ===\n"
                logger.info(f"Retrieved {len(doctrine_docs)} doctrine references")
            else:
                logger.warning("No KB documents found")
        else:
            logger.warning("No vectorstore provided - KB not accessible")
        
        # Build summaries
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
            for obj in yolo_objects[:10]:
                yolo_summary += f"  - {obj['type']} at ({obj['position'][0]}, {obj['position'][1]}) [confidence: {obj['confidence']}]\n"
        else:
            yolo_summary += "No objects detected\n"

        # Build NATO symbols summary
        symbols_summary = "\n═══════════════════════════════════════════════════════════════════\n"
        symbols_summary += "NATO MILITARY SYMBOLS DETECTED ON MAP (ResNet18 Classification)\n"
        symbols_summary += "═══════════════════════════════════════════════════════════════════\n"

        if nato_symbols:
            symbols_summary += f"\nIdentified {len(nato_symbols)} NATO symbols:\n\n"
            for idx, symbol in enumerate(nato_symbols, 1):
                conf_indicator = " ⚠️ (Low Confidence)" if symbol['low_confidence'] else ""
                symbols_summary += f"{idx}. {symbol['symbol']}\n"
                symbols_summary += f"   Location: {symbol['sector']} sector\n"
                symbols_summary += f"   Confidence: {symbol['confidence']:.1%}{conf_indicator}\n\n"

            symbols_summary += "Note: Symbols marked with ⚠️ have <85% confidence.\n"
            symbols_summary += "If any symbols appear incorrect or missing, please notify me.\n"
        else:
            symbols_summary += "\nNo NATO symbols detected on this map.\n"
            if self.symbol_classifier is None:
                symbols_summary += "(Symbol classifier not loaded - model training may be required)\n"

        symbols_summary += "═══════════════════════════════════════════════════════════════════\n"

        # PHASE 5: Strategy Synthesis
        logger.info("\n[5/5] Llama 3.2: Strategy Synthesis")
        
        strategy_prompt = f"""You are a NATO tactical operations officer conducting mission planning with access to alliance doctrine and multi-source intelligence.

You are expert in interpreting NATO APP-6(E) military symbology on tactical maps. When analyzing the visual reconnaissance, pay special attention to:
- Identified military symbols and their tactical significance
- Unit dispositions (friendly, enemy, neutral forces) and their relationship to terrain
- Echelon indicators showing unit size and command relationships
- How symbol placement reflects tactical principles from doctrine

═══════════════════════════════════════════════════════════════════
OPERATION DEVELOPMENT
═══════════════════════════════════════════════════════════════════

MISSION: {scenario}

FRIENDLY FORCES (Available):
{chr(10).join([f"- {unit}: {count} units" for unit, count in unit_types.items()])}

═══════════════════════════════════════════════════════════════════
INTELLIGENCE PREPARATION OF THE BATTLEFIELD 
═══════════════════════════════════════════════════════════════════

AI-ENHANCED TERRAIN ANALYSIS (CLIP):
{clip_summary}

OBJECT DETECTION INTELLIGENCE (YOLO):
{yolo_summary}

{symbols_summary}

VISUAL RECONNAISSANCE ASSESSMENT (LLaVA):
{llava_analysis}

NATO DOCTRINAL REFERENCES:
{kb_context}

═══════════════════════════════════════════════════════════════════
TACTICAL MISSION ANALYSIS REQUIRED
═══════════════════════════════════════════════════════════════════

Develop a comprehensive tactical plan following NATO planning methodology:

1. SITUATION ANALYSIS
   - Synthesize terrain intelligence from all sources (CLIP, YOLO, LLaVA)
   - Identify key terrain and decisive points
   - Assess enemy most likely/most dangerous courses of action (if applicable)

2. MISSION ANALYSIS
   - Apply relevant NATO doctrine to this scenario
   - Identify specified, implied, and essential tasks
   - State mission in "who, what, when, where, why" format

3. COURSE OF ACTION (COA) DEVELOPMENT
   - SCHEME OF MANEUVER: How will forces be arrayed and employed?
   - TASK ORGANIZATION: How should units be task-organized?
   - FIRE SUPPORT PLAN: Artillery positioning and fire support coordination
   - SUSTAINMENT: Logistics and casualty evacuation considerations

4. EXECUTION MATRIX
   Provide specific deployment instructions:
   - Unit-by-unit positioning with grid references/compass bearings
   - Rationale tied to terrain analysis and doctrine
   - Coordination measures (phase lines, checkpoints, boundaries)
   - Contingency plans for expected friction points

5. DOCTRINAL COMPLIANCE
   - Cite specific NATO doctrine principles applied
   - Reference chapter/section if identifiable from KB
   - Explain how doctrine informs your recommendations

DELIVER A COMPLETE, ACTIONABLE TACTICAL PLAN WITH CLEAR TIES TO INTELLIGENCE AND DOCTRINE."""

        strategy = self.llm.invoke(strategy_prompt)
        
        logger.info("="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70)
        
        return {
            'strategy': strategy,
            'clip_analysis': clip_terrain,
            'clip_regions': clip_regions,
            'yolo_detections': yolo_objects,
            'nato_symbols': nato_symbols,
            'llava_analysis': llava_analysis,
            'models_used': ['CLIP', 'YOLO', 'ResNet18', 'LLaVA', 'KB', 'Llama 3.2']
        }
    
    def create_annotated_map(self, image_path, yolo_detections, clip_regions, output_path):
        """Create map with YOLO detections and CLIP region labels"""
        try:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            
            for detection in yolo_detections:
                x1, y1, x2, y2 = detection['bbox']
                label = f"{detection['type']} ({detection['confidence']})"
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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
                        
                        x = int((j + 0.5) * (width / 3))
                        y = int((i + 0.5) * (height / 3))
                        
                        terrain = region_info['terrain'].split()[0]
                        label = f"{direction.upper()[:2]}: {terrain}"
                        
                        cv2.putText(img, label, (x - 40, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imwrite(output_path, img)
            logger.info(f"Annotated map saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Map annotation failed: {e}")
            return None