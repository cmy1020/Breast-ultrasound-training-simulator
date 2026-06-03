import Sofa.Core
import SofaRuntime


def add_scene_header(
        root: Sofa.Core.Node,
        gravity: list = [0, -9.81, 0],
        dt: float = 0.01,
        alarm_distance: float = 0.004,
        contact_distance: float = 0.002,
        use_omega6: bool = False,
):
    """Create SOFA scene header with collision pipeline."""

    # SOFA v22.06+: collision components need explicit plugin loading
    SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Algorithm")
    SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Intersection")
    SofaRuntime.importPlugin("Sofa.Component.Collision.Response.Contact")

    root.dt.value = dt
    root.gravity.value = gravity

    # Visual settings
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('VisualStyle',
                   displayFlags='hideCollisionModels hideForceFields hideBehaviorModels showVisualModels')
    root.addObject('BackgroundSetting', color=[0, 0, 0, 0])

    # Animation loop (Penalty method)
    root.addObject('DefaultAnimationLoop')

    # Collision detection pipeline
    root.addObject('DefaultPipeline',
                   name="CollisionPipeline",
                   depth=6,
                   verbose=False)

    root.addObject('BruteForceBroadPhase',
                   name="BroadPhase")

    root.addObject('BVHNarrowPhase',
                   name="NarrowPhase")

    root.addObject('MinProximityIntersection',
                   name='Proximity',
                   alarmDistance=alarm_distance,
                   contactDistance=contact_distance)

    # Contact response (Penalty method)
    contact_stiffness = 500 if use_omega6 else 30000

    if use_omega6:
        print("\n" + "=" * 60)
        print("Scene config (Penalty method):")
        print(f"  dt: {dt}s")
        print(f"  alarmDistance: {alarm_distance * 1000:.1f}mm")
        print(f"  contactDistance: {contact_distance * 1000:.1f}mm")
        print(f"  contactStiffness: {contact_stiffness} N/m")
        print(f"  Mode: Omega6 (Penalty)")
        print("=" * 60 + "\n")

    root.addObject('RuleBasedContactManager',
                   name="ContactManager",
                   response='PenalityContactForceField',
                   responseParams=f'stiffness={contact_stiffness}')

    return root
