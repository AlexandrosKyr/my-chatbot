#!/usr/bin/env python3
"""
Tactical Defense Analyzer

Analyzes map data and provides NATO doctrine-based defensive recommendations.
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class TacticalDefenseAnalyzer:
    """NATO doctrine-based defensive position analysis"""

    def __init__(self):
        """Initialize tactical analyzer"""
        self.doctrine_principles = {
            'defense': {
                'key_terrain': 'Occupy and defend key terrain that provides observation and fields of fire',
                'depth': 'Organize defense in depth with multiple defensive positions',
                'mutual_support': 'Position units to provide mutual fire support',
                'obstacle_integration': 'Integrate natural and man-made obstacles',
                'reserve': 'Maintain a mobile reserve for counterattacks'
            },
            'terrain_analysis': {
                'OCOKA': {
                    'O': 'Observation and Fields of Fire',
                    'C': 'Cover and Concealment',
                    'O2': 'Obstacles',
                    'K': 'Key Terrain',
                    'A': 'Avenues of Approach'
                }
            }
        }

    def analyze_defensive_position(self, map_analysis: Dict) -> Dict:
        """
        Analyze defensive positions based on map data

        Args:
            map_analysis: Results from integrated map analyzer

        Returns:
            Defensive analysis with recommendations
        """
        logger.info("Analyzing defensive positions based on NATO doctrine...")

        analysis = {
            'ocoka_analysis': {},
            'defensive_recommendations': [],
            'key_positions': [],
            'vulnerabilities': [],
            'strength_assessment': ''
        }

        # Extract data
        location = map_analysis.get('location', {})
        terrain = map_analysis.get('terrain_classification', {})
        topo = map_analysis.get('topographic', {})
        google = map_analysis.get('google_maps', {})

        # OCOKA Analysis
        analysis['ocoka_analysis'] = self._analyze_ocoka(terrain, topo, google)

        # Defensive recommendations
        analysis['defensive_recommendations'] = self._generate_defensive_recommendations(
            terrain, topo, google
        )

        # Key positions
        analysis['key_positions'] = self._identify_key_positions(terrain, topo)

        # Vulnerabilities
        analysis['vulnerabilities'] = self._identify_vulnerabilities(terrain, topo)

        # Overall assessment
        analysis['strength_assessment'] = self._assess_defensive_strength(terrain, topo)

        return analysis

    def _analyze_ocoka(self, terrain: Dict, topo: Dict, google: Dict) -> Dict:
        """OCOKA analysis (NATO terrain analysis methodology)"""
        ocoka = {
            'observation': '',
            'cover_concealment': '',
            'obstacles': '',
            'key_terrain': '',
            'avenues_of_approach': ''
        }

        # Observation and Fields of Fire
        if terrain and terrain.get('terrain_distribution'):
            terrain_dist = terrain['terrain_distribution']

            # High ground provides excellent observation
            if 'mountainous' in terrain_dist or 'high_ground' in terrain.get('tactical_distribution', {}):
                ocoka['observation'] = "EXCELLENT - Mountainous terrain with {0} contour lines provides commanding observation positions. Elevation advantage allows long-range target acquisition.".format(
                    topo.get('contours', {}).get('num_contours', 'numerous')
                )
            else:
                ocoka['observation'] = "MODERATE - Relatively flat terrain limits long-range observation. Consider establishing OPs on available high ground."

        # Cover and Concealment
        tactical_dist = terrain.get('tactical_distribution', {})
        if tactical_dist.get('cover', 0) > 0.3:
            cover_pct = tactical_dist['cover'] * 100
            ocoka['cover_concealment'] = f"GOOD - {cover_pct:.0f}% of area provides cover and concealment. Utilize natural features for defensive positions."
        elif topo.get('terrain_colors', {}).get('vegetation', {}).get('coverage_percent', 0) > 3:
            veg_pct = topo['terrain_colors']['vegetation']['coverage_percent']
            ocoka['cover_concealment'] = f"MODERATE - {veg_pct:.1f}% vegetation provides limited concealment. Augment with fighting positions and camouflage."
        else:
            ocoka['cover_concealment'] = "LIMITED - Open terrain provides minimal natural cover. Recommend engineer support for fortifications."

        # Obstacles
        rivers = topo.get('rivers', {}).get('num_segments', 0)
        if rivers > 5:
            ocoka['obstacles'] = f"GOOD - {rivers} river/stream segments identified as natural obstacles. Can canalize enemy movement and serve as defensive barriers."
        else:
            ocoka['obstacles'] = f"LIMITED - Minimal natural obstacles ({rivers} water features). Recommend emplacement of tactical obstacles (wire, mines, tank ditches)."

        # Key Terrain
        if 'mountainous' in terrain.get('terrain_distribution', {}):
            ocoka['key_terrain'] = "HIGH GROUND identified as key terrain. Control of elevated positions is critical for defense. Priority: occupy and fortify dominant heights."
        else:
            roads = topo.get('roads', {}).get('num_segments', 0)
            ocoka['key_terrain'] = f"ROAD NETWORK ({roads} segments) constitutes key terrain. Control of road junctions denies enemy mobility."

        # Avenues of Approach
        if tactical_dist.get('killzone', 0) > 0.3:
            kz_pct = tactical_dist['killzone'] * 100
            ocoka['avenues_of_approach'] = f"IDENTIFIABLE - {kz_pct:.0f}% open killzones indicate likely enemy approach routes. Position direct fire weapons to cover these areas."
        else:
            ocoka['avenues_of_approach'] = "MULTIPLE - Varied terrain allows multiple approach routes. Requires strong reconnaissance and security elements."

        return ocoka

    def _generate_defensive_recommendations(self, terrain: Dict, topo: Dict, google: Dict) -> List[str]:
        """Generate specific defensive recommendations"""
        recommendations = []

        terrain_dist = terrain.get('terrain_distribution', {})
        tactical_dist = terrain.get('tactical_distribution', {})

        # 1. Positioning
        if 'mountainous' in terrain_dist or tactical_dist.get('high_ground', 0) > 0.5:
            recommendations.append(
                "🏔️ MAIN DEFENSIVE POSITIONS: Establish primary defensive line on high ground. "
                "Emplace anti-tank weapons on reverse slopes for protection from direct fire."
            )

        # 2. Fire Support
        if tactical_dist.get('killzone', 0) > 0.3:
            recommendations.append(
                "🎯 ENGAGEMENT AREAS: Designate {0:.0f}% of open terrain as primary engagement areas. "
                "Pre-register indirect fire and establish final protective fires.".format(
                    tactical_dist['killzone'] * 100
                )
            )

        # 3. Obstacles
        rivers = topo.get('rivers', {}).get('num_segments', 0)
        if rivers > 0:
            recommendations.append(
                f"🚧 OBSTACLE INTEGRATION: Utilize {rivers} water obstacles to canalize enemy. "
                "Augment with engineer obstacles (wire, mines) on likely crossing sites."
            )

        # 4. Depth
        recommendations.append(
            "📏 DEFENSE IN DEPTH: Organize battle positions in depth - forward security zone, "
            "main defensive belt (MBA), and reserve positions. Minimum 2-3km depth."
        )

        # 5. Mutual Support
        roads = topo.get('roads', {}).get('num_segments', 0)
        if roads > 50:
            recommendations.append(
                f"🔗 MUTUAL SUPPORT: {roads} road segments enable lateral movement. "
                "Position units within supporting range (400-800m for small arms, 2-4km for direct fire systems)."
            )

        # 6. Reserve
        recommendations.append(
            "⚔️ RESERVE FORCE: Maintain 1/3 of force as mobile reserve for counterattacks. "
            "Position centrally with access to multiple routes for rapid deployment."
        )

        # 7. Reconnaissance
        recommendations.append(
            "👁️ RECONNAISSANCE: Emplace observation posts on high ground. "
            "Establish ground surveillance radar positions. Maintain continuous surveillance of NAIs."
        )

        return recommendations

    def _identify_key_positions(self, terrain: Dict, topo: Dict) -> List[Dict]:
        """Identify specific defensive positions"""
        positions = []

        tactical_dist = terrain.get('tactical_distribution', {})

        # High ground positions
        if tactical_dist.get('high_ground', 0) > 0.5:
            positions.append({
                'type': 'Command Post / Observation Post',
                'location': 'High ground (mountainous terrain)',
                'purpose': 'Observation, command and control',
                'priority': 'CRITICAL'
            })

            positions.append({
                'type': 'Anti-Tank Positions',
                'location': 'Reverse slopes of high ground',
                'purpose': 'Engage enemy armor with protection from direct fire',
                'priority': 'HIGH'
            })

        # Killzones
        if tactical_dist.get('killzone', 0) > 0.3:
            positions.append({
                'type': 'Engagement Area',
                'location': 'Open terrain zones',
                'purpose': 'Primary kill zones for direct and indirect fire',
                'priority': 'HIGH'
            })

        # Cover positions
        if tactical_dist.get('cover', 0) > 0.3:
            positions.append({
                'type': 'Fighting Positions',
                'location': 'Areas with natural cover',
                'purpose': 'Infantry defensive positions',
                'priority': 'MEDIUM'
            })

        return positions

    def _identify_vulnerabilities(self, terrain: Dict, topo: Dict) -> List[str]:
        """Identify defensive vulnerabilities"""
        vulnerabilities = []

        terrain_dist = terrain.get('terrain_distribution', {})
        tactical_dist = terrain.get('tactical_distribution', {})

        # Limited cover
        if tactical_dist.get('cover', 0) < 0.2:
            vulnerabilities.append(
                "⚠️ LIMITED COVER: Open terrain provides minimal protection. "
                "Vulnerable to artillery and air attack. Recommend extensive fortification."
            )

        # Flat terrain
        if 'mountainous' not in terrain_dist and tactical_dist.get('high_ground', 0) < 0.3:
            vulnerabilities.append(
                "⚠️ FLAT TERRAIN: Limited observation. Vulnerable to infiltration. "
                "Requires strong security zone and patrols."
            )

        # Few obstacles
        if topo.get('rivers', {}).get('num_segments', 0) < 3:
            vulnerabilities.append(
                "⚠️ FEW NATURAL OBSTACLES: Enemy has multiple approach routes. "
                "Engineer support critical for obstacle emplacement."
            )

        # Many roads
        if topo.get('roads', {}).get('num_segments', 0) > 80:
            vulnerabilities.append(
                "⚠️ EXTENSIVE ROAD NETWORK: Enemy has high mobility. "
                "Prepare obstacles on roads and control key intersections."
            )

        return vulnerabilities

    def _assess_defensive_strength(self, terrain: Dict, topo: Dict) -> str:
        """Overall defensive strength assessment"""
        score = 0
        factors = []

        tactical_dist = terrain.get('tactical_distribution', {})

        # High ground (+3)
        if tactical_dist.get('high_ground', 0) > 0.7:
            score += 3
            factors.append("excellent high ground")

        # Cover (+2)
        if tactical_dist.get('cover', 0) > 0.4:
            score += 2
            factors.append("good cover")

        # Obstacles (+2)
        if topo.get('rivers', {}).get('num_segments', 0) > 10:
            score += 2
            factors.append("natural obstacles")

        # Killzones (+1)
        if tactical_dist.get('killzone', 0) > 0.3:
            score += 1
            factors.append("clear fields of fire")

        # Assessment
        if score >= 6:
            return f"STRONG defensive position ({', '.join(factors)}). Suitable for battalion-level defense."
        elif score >= 4:
            return f"MODERATE defensive position ({', '.join(factors)}). Requires augmentation with obstacles and fortifications."
        else:
            return f"WEAK defensive position. Consider alternative positions or extensive engineering support."

    def format_analysis(self, analysis: Dict) -> str:
        """Format analysis as readable report"""
        report = []
        report.append("="*70)
        report.append("NATO DOCTRINE-BASED DEFENSIVE ANALYSIS")
        report.append("="*70)

        # OCOKA
        report.append("\n📋 OCOKA ANALYSIS:")
        report.append("-" * 70)
        ocoka = analysis['ocoka_analysis']
        report.append(f"\n🔭 OBSERVATION & FIELDS OF FIRE:")
        report.append(f"   {ocoka.get('observation', 'N/A')}")
        report.append(f"\n🛡️ COVER & CONCEALMENT:")
        report.append(f"   {ocoka.get('cover_concealment', 'N/A')}")
        report.append(f"\n🚧 OBSTACLES:")
        report.append(f"   {ocoka.get('obstacles', 'N/A')}")
        report.append(f"\n🎯 KEY TERRAIN:")
        report.append(f"   {ocoka.get('key_terrain', 'N/A')}")
        report.append(f"\n🛣️ AVENUES OF APPROACH:")
        report.append(f"   {ocoka.get('avenues_of_approach', 'N/A')}")

        # Recommendations
        report.append("\n\n💡 DEFENSIVE RECOMMENDATIONS:")
        report.append("-" * 70)
        for i, rec in enumerate(analysis['defensive_recommendations'], 1):
            report.append(f"\n{i}. {rec}")

        # Key Positions
        if analysis['key_positions']:
            report.append("\n\n📍 KEY DEFENSIVE POSITIONS:")
            report.append("-" * 70)
            for pos in analysis['key_positions']:
                report.append(f"\n• {pos['type']} [{pos['priority']}]")
                report.append(f"  Location: {pos['location']}")
                report.append(f"  Purpose: {pos['purpose']}")

        # Vulnerabilities
        if analysis['vulnerabilities']:
            report.append("\n\n⚠️ VULNERABILITIES:")
            report.append("-" * 70)
            for vuln in analysis['vulnerabilities']:
                report.append(f"\n{vuln}")

        # Overall Assessment
        report.append("\n\n📊 DEFENSIVE STRENGTH ASSESSMENT:")
        report.append("-" * 70)
        report.append(f"\n{analysis['strength_assessment']}")

        report.append("\n" + "="*70)

        return "\n".join(report)


def main():
    """Test tactical analyzer with sample data"""
    import json

    # Load results from integrated analyzer
    results_file = './map_analysis_results.json'
    try:
        with open(results_file, 'r') as f:
            map_analysis = json.load(f)
    except FileNotFoundError:
        print(f"Run integrated_map_analyzer.py first to generate {results_file}")
        return

    # Analyze defensive position
    analyzer = TacticalDefenseAnalyzer()
    defensive_analysis = analyzer.analyze_defensive_position(map_analysis)

    # Print report
    report = analyzer.format_analysis(defensive_analysis)
    print(report)

    # Save report
    with open('./defensive_analysis_report.txt', 'w') as f:
        f.write(report)
    print(f"\n✓ Defensive analysis saved to: defensive_analysis_report.txt")


if __name__ == "__main__":
    main()
