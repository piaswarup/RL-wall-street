"""Generate a parametric URDF for the Vertigo-style wall-climbing robot.

Layout (matches the real robot as described on 2026-11-22):
  * flat chassis with four FREE-ROLLING wheels (no drive motors, no steering)
  * two identical propeller units on top (front and rear), each on ONE continuous (360 deg) tilt joint whose
    axis runs side-to-side (body Y):

        base_link --(front_tilt, about body Y, continuous)--> front_prop
        base_link --(rear_tilt,  about body Y, continuous)--> rear_prop

  * the propellers are flipped so they blow air AWAY from the robot: at zero tilt the airflow goes up (+Z of the
    prop link) and the thrust force pushes the robot DOWN into the surface it is on.

Thrust is NOT in the URDF; the RL environment applies it to ``front_prop`` / ``rear_prop`` along their local -Z.

Edit the PARAMETERS block to match the real robot, then regenerate (see WALKTHROUGH.md section 3):

    uv run python scripts/make_vertigo_urdf.py
    uv run --extra importers python ../IsaacLab/scripts/tools/convert_urdf.py \
        src/vertigo_rl/assets/data/vertigo/vertigo.urdf src/vertigo_rl/assets/data/vertigo/usd

Units: metres, kilograms, radians.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- PARAMETERS
CHASSIS_SIZE = (0.36, 0.30, 0.06)  # length (x), width (y), thickness (z)
CHASSIS_MASS = 1.55  # incl. battery and electronics; total robot mass = 2.00 kg

WHEEL_RADIUS = 0.06
WHEEL_MASS = 0.06
WHEEL_X = 0.15  # +/- along body x (front/rear axle)
WHEEL_Y = 0.18  # +/- along body y (left/right)

PROP_X = 0.11  # +/- along body x (front/rear propeller unit)
PROP_Z = 0.07  # height of the tilt axis above chassis centre
PROP_RADIUS = 0.09  # propeller disc (visual + collision)
PROP_UNIT_MASS = 0.105  # motor + prop + tilt hardware that rotates with the prop

TILT_EFFORT = 5.0  # N*m tilt-servo torque limit
TILT_VELOCITY = 8.0  # rad/s tilt-servo speed limit (~460 deg/s) -- MEASURE THIS
# ---------------------------------------------------------------------------

OUT = Path(__file__).resolve().parents[1] / "src/vertigo_rl/assets/data/vertigo/vertigo.urdf"


def box_inertia(m, x, y, z):
    return (m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12)


def sphere_inertia(m, r):
    i = 2 * m * r * r / 5
    return (i, i, i)


def cyl_inertia(m, r, h):
    ixx = m * (3 * r * r + h * h) / 12
    return (ixx, ixx, m * r * r / 2)


def inertial(m, inertia):
    ixx, iyy, izz = inertia
    return (
        f'    <inertial><origin xyz="0 0 0"/><mass value="{m:.4f}"/>'
        f'<inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/></inertial>\n'
    )


def material(name, rgba):
    return f'      <material name="{name}"><color rgba="{rgba}"/></material>\n'


def build() -> str:
    lx, ly, lz = CHASSIS_SIZE
    s = ['<?xml version="1.0"?>\n<robot name="vertigo">\n']

    s.append('  <link name="base_link">\n')
    s.append(inertial(CHASSIS_MASS, box_inertia(CHASSIS_MASS, lx, ly, lz)))
    s.append(f'    <visual><geometry><box size="{lx} {ly} {lz}"/></geometry>\n')
    s.append(material("chassis", "0.15 0.35 0.75 1") + "    </visual>\n")
    s.append(f'    <collision><geometry><box size="{lx} {ly} {lz}"/></geometry></collision>\n')
    s.append("  </link>\n")

    for name, sx, sy in (("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)):
        s.append(f'  <link name="wheel_{name}">\n')
        s.append(inertial(WHEEL_MASS, sphere_inertia(WHEEL_MASS, WHEEL_RADIUS)))
        s.append(f'    <visual><geometry><sphere radius="{WHEEL_RADIUS}"/></geometry>\n')
        s.append(material("tire", "0.1 0.1 0.1 1") + "    </visual>\n")
        # spheres roll cleanly on both the floor and the wall and have no orientation issues
        s.append(f'    <collision><geometry><sphere radius="{WHEEL_RADIUS}"/></geometry></collision>\n')
        s.append("  </link>\n")
        s.append(f'  <joint name="wheel_{name}" type="continuous">\n')
        s.append(f'    <parent link="base_link"/><child link="wheel_{name}"/>\n')
        s.append(f'    <origin xyz="{sx * WHEEL_X} {sy * WHEEL_Y} 0"/><axis xyz="0 1 0"/>\n')
        s.append('    <limit effort="0.0" velocity="100.0"/>\n  </joint>\n')

    for unit, sx in (("front", 1), ("rear", -1)):
        s.append(f'  <link name="{unit}_prop">\n')
        s.append(inertial(PROP_UNIT_MASS, cyl_inertia(PROP_UNIT_MASS, 0.03, 0.04)))
        # thin disc = propeller plane
        s.append(f'    <visual><geometry><cylinder radius="{PROP_RADIUS}" length="0.005"/></geometry>\n')
        colour = "0.9 0.4 0.1 0.6" if unit == "front" else "0.2 0.8 0.3 0.6"
        s.append(material(f"{unit}_prop", colour) + "    </visual>\n")
        # solid prop disc: the robot must not be able to rest on (or drive with) its propellers against a surface
        s.append(f'    <collision><geometry><cylinder radius="{PROP_RADIUS}" length="0.02"/></geometry></collision>\n')
        # red stick = direction the AIR is blown (+Z); the force on the robot is the opposite way
        s.append('    <visual><origin xyz="0 0 0.05"/><geometry><cylinder radius="0.006" length="0.10"/></geometry>\n')
        s.append(material("airflow", "1 0 0 1") + "    </visual>\n  </link>\n")
        s.append(f'  <joint name="{unit}_tilt" type="continuous">\n')
        s.append(f'    <parent link="base_link"/><child link="{unit}_prop"/>\n')
        s.append(f'    <origin xyz="{sx * PROP_X} 0 {PROP_Z}"/><axis xyz="0 1 0"/>\n')
        s.append(f'    <limit effort="{TILT_EFFORT}" velocity="{TILT_VELOCITY}"/>\n  </joint>\n')

    s.append("</robot>\n")
    return "".join(s)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    total = CHASSIS_MASS + 4 * WHEEL_MASS + 2 * PROP_UNIT_MASS
    print(f"Wrote {OUT}  (total mass {total:.3f} kg, weight {total * 9.81:.2f} N)")
