"""
This script loads a 3D model, normalizes its position and scale, sets up
lighting, and renders it from multiple camera angles. It supports lighting via
an environment map or by making materials emissive (baked lighting).
It can render images with filenames that encode the camera's azimuth and elevation.
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

try:
    import bpy
    from mathutils import Matrix, Vector
    from bpy_extras.io_utils import axis_conversion
except ImportError:
    print("This script must be run within Blender's Python environment.")
    sys.exit(1)

_CONTEXT = bpy.context
_SCENE   = _CONTEXT.scene
_RENDER  = _SCENE.render

# A dictionary mapping file extensions to Blender's import functions.
IMPORT_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "obj":  bpy.ops.wm.obj_import,
    "glb":  bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd":  bpy.ops.import_scene.usd,
    "fbx":  bpy.ops.import_scene.fbx,
    "stl":  bpy.ops.import_mesh.stl,
    "ply":  bpy.ops.import_mesh.ply,
}

# Map for translating user-friendly arguments to Blender's enum identifiers.
AXIS_MAP = {
    'X': 'X', 'Y': 'Y', 'Z': 'Z',
    '-X': 'NEGATIVE_X', '-Y': 'NEGATIVE_Y', '-Z': 'NEGATIVE_Z'
}

@dataclass(frozen=True)
class RenderTask:
    """An immutable container for defining a single rendering job."""
    num_renders: int
    random_camera: bool
    camera_mode: str
    elevations: Tuple[float, ...]
    min_elev: float

def reset_scene() -> None:
    """Resets the Blender scene to a clean state by removing all but essential objects."""
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            obj.select_set(True)
    if _CONTEXT.selected_objects:
        bpy.ops.object.delete()

    # Comprehensive cleanup of data-blocks to avoid inter-run contamination.
    for collection in [
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.meshes,
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.node_groups,
    ]:
        while collection:
            collection.remove(collection[0])

def clear_all_animation_data() -> None:
    print("Clearing all animation data to ensure static geometry...")
    for obj in bpy.data.objects:
        obj.animation_data_clear()
    if _SCENE.animation_data:
        _SCENE.animation_data_clear()

def load_object(object_path: Path, axis_forward: str, axis_up: str) -> None:
    """Loads a model with robust, format-aware axis correction."""
    if not object_path.is_file():
        raise FileNotFoundError(f"Object file not found: {object_path}")

    file_extension = object_path.suffix.lstrip(".").lower()
    import_func = IMPORT_FUNCTIONS.get(file_extension)
    if not import_func:
        raise ValueError(f"Unsupported file extension: '{file_extension}'")

    kwargs = {}
    final_correction_matrix = Matrix.Identity(4)

    match file_extension:
        case "obj" | "fbx" | "stl":
            kwargs['forward_axis'] = AXIS_MAP[axis_forward]
            kwargs['up_axis'] = AXIS_MAP[axis_up]

        case "glb" | "gltf":
            kwargs["merge_vertices"] = True
            target_correction = axis_conversion(
                from_forward=axis_forward, from_up=axis_up,
                to_forward='Y', to_up='Z'
            ).to_4x4()
            # Undo Blender's implicit +90deg X-rotation for glTF imports.
            gltf_undo_matrix = Matrix.Rotation(math.radians(-90.0), 4, 'X')
            # Apply undo matrix first, then the target correction.
            final_correction_matrix = target_correction @ gltf_undo_matrix

        case _:  # PLY, USD, etc.
            final_correction_matrix = axis_conversion(
                from_forward=axis_forward, from_up=axis_up,
                to_forward='Y', to_up='Z'
            ).to_4x4()

    objects_before = set(bpy.context.scene.objects)
    import_func(filepath=str(object_path), **kwargs)

    if not final_correction_matrix.is_identity:
        new_objects = set(bpy.context.scene.objects) - objects_before
        new_root_objects = [o for o in new_objects if o.parent is None or o.parent not in new_objects]

        for obj in new_root_objects:
            obj.matrix_world = final_correction_matrix @ obj.matrix_world

def get_scene_aabb() -> Tuple[Vector, Vector]:
    """
    Computes the axis-aligned bounding box (AABB) of all mesh objects in the scene.

    Returns:
        A tuple containing the minimum and maximum corner vectors of the AABB.

    Raises:
        RuntimeError: If no mesh objects are found in the scene.
    """
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    found_mesh = False

    for obj in _SCENE.objects:
        if obj.type == "MESH":
            found_mesh = True
            for vertex in obj.data.vertices:
                global_vertex = obj.matrix_world @ vertex.co
                for i in range(3):
                    bbox_min[i] = min(bbox_min[i], global_vertex[i])
                    bbox_max[i] = max(bbox_max[i], global_vertex[i])

    if not found_mesh:
        raise RuntimeError("No mesh objects in scene to compute AABB for.")

    return bbox_min, bbox_max


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Yields all root objects (objects with no parent) in the scene."""
    for obj in _SCENE.objects:
        if not obj.parent:
            yield obj


def normalize_scene() -> None:
    """Normalizes the scene to fit in a unit cube and applies type-specific rotations."""
    root_objects = [obj for obj in get_scene_root_objects() if obj.type != "CAMERA"]
    if not root_objects:
        print("Warning: No objects to normalize.")
        return

    transform_target = root_objects[0]
    if len(root_objects) > 1:
        parent_empty = bpy.data.objects.new("NormalizationParent", None)
        _SCENE.collection.objects.link(parent_empty)
        for obj in root_objects:
            obj.parent = parent_empty
        transform_target = parent_empty

    _CONTEXT.view_layer.update()

    try:
        bbox_min, bbox_max = get_scene_aabb()
    except RuntimeError:
        print("Warning: No meshes in scene. Skipping normalization.")
        return

    scale_factor = 1.0 / max((bbox_max - bbox_min).length, 1e-6)
    offset = -(bbox_min + bbox_max) / 2.0

    transform_target.scale = Vector((scale_factor, scale_factor, scale_factor))
    transform_target.location = offset * scale_factor
    _CONTEXT.view_layer.update()

def set_lighting(
    env_map_path: Optional[Path], is_baked: bool, use_emission_shader: bool
) -> None:
    """Configures the scene's lighting based on provided arguments."""
    if is_baked:
        print("Using baked emission shader (vertex colors).")
        set_emission_shader_from_vertex_color()
    elif use_emission_shader:
        print("Using emission shader (texture/albedo).")
        set_emission_shader_from_texture()
    elif env_map_path:
        print("Using environment map lighting.")
        load_environment_map(env_map_path)
    else:
        print("Warning: No lighting specified. Scene may be dark.")


def load_environment_map(env_map_path: Path) -> None:
    """Loads an HDR environment map and sets it as the world background."""
    if not env_map_path.exists():
        print(f"Warning: Environment map not found at {env_map_path}. Skipping.")
        return

    world = _SCENE.world or bpy.data.worlds.new("World")
    _SCENE.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    bg_node = nodes.new(type="ShaderNodeBackground")
    env_tex_node = nodes.new(type="ShaderNodeTexEnvironment")
    output_node = nodes.new(type="ShaderNodeOutputWorld")

    env_tex_node.image = bpy.data.images.load(str(env_map_path))
    links = world.node_tree.links
    links.new(env_tex_node.outputs["Color"], bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def _replace_material_with_emission(obj: bpy.types.Object, setup_nodes: Callable):
    """Helper to replace materials on an object with a new emission setup."""
    mat = bpy.data.materials.new(name=f"{obj.name}_Emission")
    mat.use_nodes = True
    setup_nodes(mat.node_tree)

    if obj.material_slots:
        for slot in obj.material_slots:
            slot.material = mat
    else:
        obj.data.materials.append(mat)


def set_emission_shader_from_texture() -> None:
    """Replaces all materials with an emission shader using the base texture."""
    for obj in _SCENE.objects:
        if obj.type != "MESH":
            continue

        source_image = None
        for mat_slot in obj.material_slots:
            if mat := mat_slot.material:
                if mat.use_nodes and (
                    img_node := next(
                        (n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"), None
                    )
                ) and img_node.image:
                    source_image = img_node.image
                    break

        def setup_texture_emission(node_tree):
            nodes, links = node_tree.nodes, node_tree.links
            nodes.clear()
            tex_node = None
            if source_image:
                tex_node = nodes.new(type="ShaderNodeTexImage")
                tex_node.image = source_image
            else:
                tex_node = nodes.new(type="ShaderNodeRGB")
                tex_node.outputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
            emission = nodes.new(type="ShaderNodeEmission")
            output = nodes.new(type="ShaderNodeOutputMaterial")
            links.new(tex_node.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

        _replace_material_with_emission(obj, setup_texture_emission)


def set_emission_shader_from_vertex_color() -> None:
    """Replaces all materials with an emission shader that uses vertex colors."""
    for obj in _SCENE.objects:
        if (
            obj.type != "MESH"
            or not obj.data.color_attributes
            or not (color_layer := obj.data.color_attributes.active)
        ):
            continue

        def setup_vcolor_emission(node_tree):
            nodes, links = node_tree.nodes, node_tree.links
            nodes.clear()
            attr_node = nodes.new(type="ShaderNodeAttribute")
            attr_node.attribute_name = color_layer.name
            emission = nodes.new(type="ShaderNodeEmission")
            output = nodes.new(type="ShaderNodeOutputMaterial")
            links.new(attr_node.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

        _replace_material_with_emission(obj, setup_vcolor_emission)


def get_camera_positions(
    task: RenderTask, radius: float
) -> Generator[Tuple[Vector, float, float], None, None]:
    """
    Generates camera positions using direct spherical coordinate calculation
    that aligns with Blender's world coordinate system (Right-Handed, Z-up):

        +Z (Up)
        ^    
        |    / +Y (Away/Forward)
        |   /
        |  /
        | /
        +------------> +X (Right)

    This function calculates points on a sphere around the origin (0,0,0)
    in this coordinate system. The negative sign in the `y` calculation
    is what places the camera in front of the object (looking from the
    positive Y direction towards the origin) for a 0-degree azimuth.
    """
    if task.camera_mode == "random":
        for _ in range(task.num_renders):
            azimuth_rad = random.uniform(0, 2 * math.pi)
            elevation_rad = math.asin(random.uniform(-1, 1))

            x = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
            y = -radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
            z = radius * math.sin(elevation_rad)

            yield Vector((x, y, z)), math.degrees(azimuth_rad), math.degrees(elevation_rad)

    else:  # Orbit / multi-orbit
        elevations = task.elevations or (0.0,)
        clamped_elevations = [max(elev, task.min_elev) for elev in elevations]
        total_renders = max(task.num_renders, len(clamped_elevations))
        base_count = max(total_renders // len(clamped_elevations), 1)
        remainder = total_renders - base_count * len(clamped_elevations)

        for ring_index, elevation_deg in enumerate(clamped_elevations):
            ring_count = base_count + (1 if ring_index < remainder else 0)
            elevation_rad = math.radians(elevation_deg)
            for i in range(ring_count):
                azimuth_deg = (360.0 / ring_count) * i
                azimuth_rad = math.radians(azimuth_deg)

                x = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
                y = -radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
                z = radius * math.sin(elevation_rad)

                yield Vector((x, y, z)), azimuth_deg, elevation_deg


def setup_camera_and_track(location: Vector, fov_deg: float) -> bpy.types.Object:
    """Positions and configures the scene camera, and makes it track the origin."""
    bpy.ops.object.camera_add(location=location)
    cam = _CONTEXT.active_object
    cam.name = "SceneCamera"
    _SCENE.camera = cam

    cam.data.type = "PERSP"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.angle = math.radians(fov_deg)

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    target = _CONTEXT.active_object
    target.name = "TrackTarget"

    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    return cam


def compute_intrinsics_from_hfov(
    fov_deg: float, width: int, height: int
) -> Tuple[float, float, float, float]:
    """Compute focal length and principal point from a horizontal field of view."""
    fov_rad = math.radians(fov_deg)
    fl_x = 0.5 * width / math.tan(0.5 * fov_rad)
    fov_y = 2.0 * math.atan(math.tan(0.5 * fov_rad) * (height / width))
    fl_y = 0.5 * height / math.tan(0.5 * fov_y)
    cx = width * 0.5
    cy = height * 0.5
    return fl_x, fl_y, cx, cy


def matrix_to_list(matrix: Matrix) -> List[List[float]]:
    """Convert a Blender Matrix to a JSON-serializable list of lists."""
    return [list(map(float, row)) for row in matrix]


def parse_elevations(elevations: str) -> Tuple[float, ...]:
    """Parse a comma-separated list of elevations in degrees."""
    if not elevations:
        return ()
    values = []
    for entry in elevations.split(","):
        entry = entry.strip()
        if not entry:
            continue
        values.append(float(entry))
    return tuple(values)


def setup_render_settings(engine: str, resolution: int, samples: int) -> None:
    """Configures global render settings for Blender."""
    _RENDER.engine = engine
    _RENDER.image_settings.file_format = "PNG"
    _RENDER.image_settings.color_mode = "RGBA"
    _RENDER.resolution_x = resolution
    _RENDER.resolution_y = resolution
    _RENDER.film_transparent = True

    if engine == "CYCLES":
        _SCENE.cycles.device = "GPU"
        _SCENE.cycles.samples = samples
        _SCENE.cycles.use_denoising = True
        _SCENE.cycles.use_persistent_data = True
        try:
            prefs = _CONTEXT.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"  # Or "OPTIX", "HIP", "METAL"
            prefs.get_devices()
            for device in prefs.devices:
                device.use = "CPU" not in device.name.upper()
        except Exception as e:
            print(f"Could not configure GPU devices, using default. Error: {e}")


def execute_render_pass(
    tasks: List[RenderTask],
    output_dir: Path,
    radius: float,
    fov_deg: float,
    camera_model: str,
) -> None:
    """
    Executes the rendering pass for a list of tasks.

    Args:
        tasks: A list of RenderTask objects defining the renders.
        output_dir: The base directory for all output renders.
        radius: The camera distance from the origin.
    """
    if not tasks:
        return

    print("\n--- Starting render pass ---")

    cam = setup_camera_and_track(Vector((0, 0, 0)), fov_deg)
    frame_mappings: Dict[int, Path] = {}
    frame_exports: Dict[Path, List[Dict[str, Any]]] = {}
    current_frame = 1

    # This loop will typically only run once, but handles the list structure.
    for task in tasks:
        if task.num_renders == 0:
            continue

        if task.camera_mode == "random":
            task_output_dir = output_dir
            prefix = "train"
            print(f"  - Planning {task.num_renders} random keyframes.")
        else:  # Orbit / multi-orbit
            task_output_dir = output_dir / "orbit"
            prefix = "orbit"
            print(
                f"  - Planning {task.num_renders} multi-orbit keyframes "
                f"across elevations {task.elevations}."
            )

        task_output_dir.mkdir(parents=True, exist_ok=True)

        pos_generator = get_camera_positions(task, radius)
        for i, (location, azimuth, elevation) in enumerate(pos_generator):
            cam.location = location
            cam.keyframe_insert(data_path="location", frame=current_frame)
            if task.camera_mode == "random":
                filename = f"{prefix}_{azimuth:.1f}_{elevation:.1f}.png"
            else:
                filename = f"{prefix}_e{int(round(elevation)):02d}_{i:03d}.png"
            frame_mappings[current_frame] = task_output_dir / filename
            current_frame += 1

    if not frame_mappings:
        return

    _SCENE.frame_start = 1
    _SCENE.frame_end = current_frame - 1

    total_frames = len(frame_mappings)
    print(f"Rendering {total_frames} frames via manual loop...")

    width = _RENDER.resolution_x
    height = _RENDER.resolution_y
    fl_x, fl_y, cx, cy = compute_intrinsics_from_hfov(fov_deg, width, height)
    opencv_conversion = Matrix(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    for frame in range(_SCENE.frame_start, _SCENE.frame_end + 1):
        _SCENE.frame_set(frame)
        output_path = frame_mappings[frame]
        transform_matrix = cam.matrix_world.copy()
        if camera_model == "OPENCV":
            transform_matrix = transform_matrix @ opencv_conversion
        frame_exports.setdefault(output_path.parent, []).append(
            {
                "file_path": output_path.name,
                "transform_matrix": matrix_to_list(transform_matrix),
            }
        )
        _RENDER.filepath = str(output_path)
        print(f"Rendering frame {frame}/{total_frames} to {output_path.name}")
        bpy.ops.render.render(write_still=True)

    print("Manual rendering complete.")

    for export_dir, frames in frame_exports.items():
        frames_sorted = sorted(frames, key=lambda entry: entry["file_path"])
        transforms_path = export_dir / "transforms.json"
        if camera_model == "OPENCV":
            payload = {
                "camera_model": "OPENCV",
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "w": width,
                "h": height,
                "frames": frames_sorted,
            }
        else:
            payload = {
                "camera_angle_x": math.radians(fov_deg),
                "w": width,
                "h": height,
                "frames": frames_sorted,
            }
        with transforms_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved transforms.json to {transforms_path} (camera_model={camera_model})")
        print(
            "Camera intrinsics: "
            f"fl_x={fl_x:.2f}, fl_y={fl_y:.2f}, cx={cx:.2f}, cy={cy:.2f}, "
            f"w={width}, h={height}"
        )
        if frames_sorted:
            first_matrix = frames_sorted[0]["transform_matrix"]
            translation = [first_matrix[i][3] for i in range(3)]
            print(f"First frame c2w translation: {translation}")

    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[cam.name].select_set(True)
    bpy.data.objects["TrackTarget"].select_set(True)
    bpy.ops.object.delete()


def main():
    """Main execution function to parse arguments and orchestrate the rendering process."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render a 3D object from multiple angles.")
    parser.add_argument("--object_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    axis_choices = ["X", "Y", "Z", "-X", "-Y", "-Z"]
    parser.add_argument(
        "--axis_forward", type=str, default='-Z', choices=axis_choices,
        help="Forward axis of the source file. Default is '-Z'."
    )
    parser.add_argument(
        "--axis_up", type=str, default='Y', choices=axis_choices,
        help="Up axis of the source file. Default is 'Y'."
    )
    parser.add_argument("--env_map_path", type=Path, default=None)
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--num_renders", type=int, default=100)
    parser.add_argument("--use_emission_shader", action="store_true")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument(
        "--fov_deg",
        type=float,
        default=60.0,
        help="Horizontal field of view in degrees for the perspective camera.",
    )
    parser.add_argument(
        "--camera_mode",
        type=str,
        choices=["random", "multi_orbit"],
        default=None,
        help="Camera sampling mode. Defaults to random if --random_camera is set; "
        "otherwise uses multi_orbit.",
    )
    parser.add_argument(
        "--elevations",
        type=str,
        default="10,25,40,55",
        help="Comma-separated list of elevation angles (degrees) for multi-orbit sampling.",
    )
    parser.add_argument(
        "--min_elev",
        type=float,
        default=5.0,
        help="Minimum elevation (degrees) to avoid below-horizon views in multi-orbit.",
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        choices=["BLENDER", "OPENCV"],
        default="BLENDER",
        help="Camera model for transforms.json export.",
    )
    parser.add_argument("--random_camera", action="store_true")
    parser.add_argument("--baked", action="store_true")
    args = parser.parse_args(argv)

    reset_scene()
    print(f"Emission shader enabled: {args.use_emission_shader}")
    load_object(args.object_path, args.axis_forward, args.axis_up)
    set_lighting(args.env_map_path, args.baked, args.use_emission_shader)
    normalize_scene()

    bpy.ops.object.select_all(action="DESELECT")
    for obj in _SCENE.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    if _CONTEXT.selected_objects:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    clear_all_animation_data()

    setup_render_settings(args.engine, args.resolution, args.samples)

    tasks: List[RenderTask] = []
    if args.num_renders > 0:
        camera_mode = args.camera_mode
        if camera_mode is None:
            camera_mode = "random" if args.random_camera else "multi_orbit"
        elevations = parse_elevations(args.elevations)
        tasks.append(
            RenderTask(
                args.num_renders,
                args.random_camera,
                camera_mode,
                elevations,
                args.min_elev,
            )
        )

    execute_render_pass(
        tasks,
        args.output_dir,
        args.radius,
        args.fov_deg,
        args.camera_model,
    )

    bpy.ops.object.select_all(action="DESELECT")
    for obj in _SCENE.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    if _CONTEXT.selected_objects and args.num_renders > 0:
        obj_name = args.object_path.stem
        normalized_path = args.output_dir / f"{obj_name}_normalized.obj"
        # WARN: Changing to coord. system of the refinement framework
        bpy.ops.export_scene.obj(
            filepath=str(normalized_path),
            use_selection=True,
            axis_forward='Z',
            axis_up='Y'
        )
        print(f"\nSaved final normalized model to {normalized_path}")
    print("Rendering complete.")

if __name__ == "__main__":
    main()
