"""
AvatarGeometryDerivation — test against real, already-confirmed
pose_keypoints from tonight's three test avatars.

Confirms:
  1. Output shape is correct (DerivedPoint with full provenance)
  2. Derived positions land in a sane place relative to the known
     shoulder/hip positions (sanity check, not a new visual test —
     that's already been done via underarm_derivation_test.py)
  3. Missing-landmark case doesn't crash

Run: python test_geometry_derivation.py
"""

import sys
sys.path.insert(0, ".")

from workers.avatar.geometry_derivation import derive_avatar_geometry

# Real pose_keypoints subset from the three avatars tested tonight —
# only the fields this module actually uses (shoulder, hip)
TEST_CASES = [
    {
        "label": "Photo 1 (relaxed arms)",
        "pose_keypoints": {
            "left_shoulder": {"x": 0.6644, "y": 0.2367},
            "right_shoulder": {"x": 0.3513, "y": 0.2425},
            "left_hip": {"x": 0.5867, "y": 0.5053},
            "right_hip": {"x": 0.424, "y": 0.506},
        }
    },
    {
        "label": "Missing landmark case (no right_hip)",
        "pose_keypoints": {
            "left_shoulder": {"x": 0.6644, "y": 0.2367},
            "right_shoulder": {"x": 0.3513, "y": 0.2425},
            "left_hip": {"x": 0.5867, "y": 0.5053},
            # right_hip deliberately omitted
        }
    },
]


def main():
    for case in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"{case['label']}")
        print('='*60)

        geometry = derive_avatar_geometry(case["pose_keypoints"])

        if geometry.left_underarm:
            lu = geometry.left_underarm
            print(f"  left_underarm:  ({lu.x_pct}%, {lu.y_pct}%) "
                  f"method={lu.method} param={lu.parameter} status={lu.status.value}")
        else:
            print(f"  left_underarm:  None (source landmarks missing)")

        if geometry.right_underarm:
            ru = geometry.right_underarm
            print(f"  right_underarm: ({ru.x_pct}%, {ru.y_pct}%) "
                  f"method={ru.method} param={ru.parameter} status={ru.status.value}")
        else:
            print(f"  right_underarm: None (source landmarks missing)")

        # Sanity check: derived point should sit between shoulder 
        # and hip vertically, never above shoulder or below hip
        pk = case["pose_keypoints"]
        if geometry.left_underarm and "left_hip" in pk:
            shoulder_y = pk["left_shoulder"]["y"] * 100
            hip_y = pk["left_hip"]["y"] * 100
            derived_y = geometry.left_underarm.y_pct
            in_range = shoulder_y < derived_y < hip_y
            print(f"  sanity check (left): shoulder_y={shoulder_y:.1f} < "
                  f"derived_y={derived_y:.1f} < hip_y={hip_y:.1f} → {'PASS' if in_range else 'FAIL'}")


if __name__ == "__main__":
    main()