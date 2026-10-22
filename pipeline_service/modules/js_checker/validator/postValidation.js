/**
 * Post-execution validation: walk the returned root, check every metric.
 *
 * Mirrors Runtime Spec § Post-Execution Validation. Single-pass bounded walk
 * with explicit cycle detection (Three.js's `Object3D.traverse` does not
 * detect cycles).
 */

import * as THREE from 'three';

export const LIMITS = {
  vertices: 250_000,
  drawCalls: 200,
  depth: 32,
  instances: 50_000,
  textureBytes: 4 * 1_048_576,
};

const ALLOWED_ROOT_TYPES = new Set([
  'Group',
  'Mesh',
  'LineSegments',
  'Points',
]);

// Standard PBR map slots that may reference a DataTexture.
const MAP_PROPS = [
  'map',
  'normalMap',
  'roughnessMap',
  'metalnessMap',
  'aoMap',
  'emissiveMap',
  'displacementMap',
  'bumpMap',
  'alphaMap',
  'envMap',
  'lightMap',
  'specularIntensityMap',
  'specularColorMap',
  'sheenRoughnessMap',
  'sheenColorMap',
  'transmissionMap',
  'thicknessMap',
  'clearcoatMap',
  'clearcoatNormalMap',
  'clearcoatRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
];


// Curved surfaces large enough to show facets must be resolved smoothly.
// Opt-in via SEGMENT_FLOOR_CHECK=1 so the rule can be A/B'd against a control.
//
// Deliberately NOT a flat floor: ~2/3 of curved primitives in real modules are
// sub-2%-of-frame details where a low segment count is correct and 32 segments
// is wasted geometry. Only primitives at or above MIN_RADIUS are checked, in
// world space (i.e. after the module's own fit-to-unit-cube normalization),
// so the threshold means the same thing in every module.
//
// Each entry pairs a segment parameter with the radius IT actually resolves.
// Torus is the trap: `radialSegments` resolves the small tube cross-section
// while the large ring is resolved by `tubularSegments`.
const SEGMENT_FLOOR_MIN_RADIUS = 0.15;

const SEGMENT_FLOORS = {
  CylinderGeometry: [['radialSegments', p => Math.max(p.radiusTop ?? 0, p.radiusBottom ?? 0), 32]],
  ConeGeometry: [['radialSegments', p => p.radius ?? 0, 32]],
  CircleGeometry: [['segments', p => p.radius ?? 0, 32]],
  SphereGeometry: [['widthSegments', p => p.radius ?? 0, 32]],
  RingGeometry: [['thetaSegments', p => p.outerRadius ?? 0, 32]],
  TorusGeometry: [
    ['radialSegments', p => p.tube ?? 0, 16],
    ['tubularSegments', p => p.radius ?? 0, 32],
  ],
  TubeGeometry: [['radialSegments', p => p.radius ?? 0, 12]],
};

const _scratchScale = new THREE.Vector3();

function collectSegmentFloorViolations(node, out) {
  const geom = node.geometry;
  const spec = geom && SEGMENT_FLOORS[geom.type];
  const params = geom && geom.parameters;
  if (!spec || !params) return;
  node.getWorldScale(_scratchScale);
  const scale = Math.max(_scratchScale.x, _scratchScale.y, _scratchScale.z);
  for (const [param, radiusOf, floor] of spec) {
    const segs = params[param];
    const worldRadius = radiusOf(params) * scale;
    if (typeof segs !== 'number' || !(worldRadius >= SEGMENT_FLOOR_MIN_RADIUS)) continue;
    if (segs < floor) {
      out.push(
        `${node.name || geom.type} (${geom.type}.${param}=${segs}, radius ${worldRadius.toFixed(2)}) needs >= ${floor}`,
      );
    }
  }
}

export function postValidate(root, options = {}) {
  const failures = [];

  if (root === null || root === undefined) {
    failures.push({
      stage: 'post_validation',
      rule: 'INVALID_RETURN_TYPE',
      detail: 'returned null or undefined',
    });
    return { failures, metrics: null };
  }

  if (!root.isObject3D) {
    failures.push({
      stage: 'post_validation',
      rule: 'INVALID_RETURN_TYPE',
      detail: `returned ${typeof root}, not an Object3D`,
    });
    return { failures, metrics: null };
  }

  if (!ALLOWED_ROOT_TYPES.has(root.type)) {
    failures.push({
      stage: 'post_validation',
      rule: 'INVALID_RETURN_TYPE',
      detail: `returned ${root.type}, expected Group/Mesh/LineSegments/Points`,
    });
    return { failures, metrics: null };
  }

  // Auto-wrap bare Mesh / LineSegments / Points returns in a Group, matching
  // Output Spec § Return Value: "Single Mesh, LineSegments, or Points returns
  // are automatically wrapped in a Group by the evaluator."
  //
  // The wrap affects depth measurement (a bare Mesh now reports maxDepth 1
  // instead of 0) but does not change vertex count, draw calls, instance
  // count, texture data, or the bounding box — the Group wrapper has an
  // identity transform and contributes nothing of its own.
  let walkRoot = root;
  if (root.type !== 'Group') {
    walkRoot = new THREE.Group();
    walkRoot.add(root);
  }

  // Bounded walk with cycle detection.
  const visited = new Set();
  const seenTextures = new Set();
  let vertices = 0;
  let drawCalls = 0;
  let maxDepth = 0;
  let instances = 0;
  let textureBytes = 0;
  let cycleDetected = false;
  let depthExceeded = false;
  // Read from options, never from `process`: this runs inside the sandbox,
  // where `process` is a trapped global and touching it throws.
  const segmentFloorCheck = options.segmentFloorCheck === true;
  const segmentFloorViolations = [];
  // World scales are read below, so make sure the matrices are current.
  walkRoot.updateMatrixWorld(true);

  function walk(node, depth) {
    if (visited.has(node)) {
      cycleDetected = true;
      return;
    }
    if (depth > LIMITS.depth) {
      depthExceeded = true;
      return;
    }
    visited.add(node);
    if (depth > maxDepth) maxDepth = depth;

    if (node.isMesh || node.isLine || node.isPoints) {
      drawCalls++;
      if (segmentFloorCheck && node.isMesh) {
        collectSegmentFloorViolations(node, segmentFloorViolations);
      }

      const geom = node.geometry;
      if (geom && geom.attributes && geom.attributes.position) {
        vertices += geom.attributes.position.count;
      }

      if (node.isInstancedMesh) {
        instances += node.count ?? 0;
      }

      const materials = Array.isArray(node.material)
        ? node.material
        : [node.material];
      for (const mat of materials) {
        if (!mat) continue;
        for (const key of MAP_PROPS) {
          const tex = mat[key];
          if (tex && tex.isDataTexture && !seenTextures.has(tex)) {
            seenTextures.add(tex);
            const data = tex.image && tex.image.data;
            if (data && typeof data.byteLength === 'number') {
              textureBytes += data.byteLength;
            }
          }
        }
      }
    }

    if (node.children) {
      for (const child of node.children) walk(child, depth + 1);
    }
  }

  walk(walkRoot, 0);

  if (cycleDetected) {
    failures.push({
      stage: 'post_validation',
      rule: 'CYCLE_DETECTED',
      detail: 'object hierarchy contains a cycle',
    });
    // Box3.setFromObject uses Three.js's traverse(), which has no cycle
    // detection and would infinitely recurse.  Return immediately.
    return {
      failures,
      metrics: { vertices, drawCalls, maxDepth, instances, textureBytes, bbox: null },
    };
  }
  if (depthExceeded) {
    failures.push({
      stage: 'post_validation',
      rule: 'DEPTH_LIMIT_EXCEEDED',
      detail: `walk exceeded max depth ${LIMITS.depth}`,
    });
  }

  if (vertices > LIMITS.vertices) {
    failures.push({
      stage: 'post_validation',
      rule: 'VERTEX_LIMIT_EXCEEDED',
      detail: `${vertices} vertices (limit ${LIMITS.vertices})`,
    });
  }
  if (drawCalls > LIMITS.drawCalls) {
    failures.push({
      stage: 'post_validation',
      rule: 'DRAW_CALL_LIMIT_EXCEEDED',
      detail: `${drawCalls} draw calls (limit ${LIMITS.drawCalls})`,
    });
  }
  if (maxDepth > LIMITS.depth) {
    failures.push({
      stage: 'post_validation',
      rule: 'DEPTH_LIMIT_EXCEEDED',
      detail: `max depth ${maxDepth} (limit ${LIMITS.depth})`,
    });
  }
  if (instances > LIMITS.instances) {
    failures.push({
      stage: 'post_validation',
      rule: 'INSTANCE_LIMIT_EXCEEDED',
      detail: `${instances} instances (limit ${LIMITS.instances})`,
    });
  }
  if (textureBytes > LIMITS.textureBytes) {
    failures.push({
      stage: 'post_validation',
      rule: 'TEXTURE_DATA_EXCEEDED',
      detail: `${textureBytes} bytes (limit ${LIMITS.textureBytes})`,
    });
  }

  // Bounding box — must fit inside [-0.5, 0.5] on all axes.
  // Computed from walkRoot (the post-wrap root) so the wrap's identity
  // transform is correctly applied — though for an identity-wrapped Mesh
  // the result is identical to setFromObject(root).
  let bbox = null;
  const box = new THREE.Box3().setFromObject(walkRoot);
  if (box.isEmpty()) {
    failures.push({
      stage: 'post_validation',
      rule: 'EMPTY_SCENE',
      detail: 'Box3.setFromObject(root).isEmpty() === true',
    });
  } else {
    bbox = {
      min: { x: box.min.x, y: box.min.y, z: box.min.z },
      max: { x: box.max.x, y: box.max.y, z: box.max.z },
    };
    if (
      box.min.x < -0.5 ||
      box.max.x > 0.5 ||
      box.min.y < -0.5 ||
      box.max.y > 0.5 ||
      box.min.z < -0.5 ||
      box.max.z > 0.5
    ) {
      failures.push({
        stage: 'post_validation',
        rule: 'BOUNDING_BOX_OUT_OF_RANGE',
        detail: `box [${fmt(box.min.x)}, ${fmt(box.min.y)}, ${fmt(box.min.z)}] to [${fmt(box.max.x)}, ${fmt(box.max.y)}, ${fmt(box.max.z)}]`,
      });
    }
  }

  if (segmentFloorViolations.length > 0) {
    failures.push({
      stage: 'post_validation',
      rule: 'CURVED_SURFACE_TOO_COARSE',
      detail:
        `${segmentFloorViolations.length} large curved surface(s) are visibly ` +
        `polygonal — raise only these segment counts, leave small details alone: ` +
        segmentFloorViolations.slice(0, 12).join('; '),
    });
  }

  return {
    failures,
    metrics: {
      vertices,
      drawCalls,
      maxDepth,
      instances,
      textureBytes,
      bbox,
    },
  };
}

function fmt(n) {
  return n.toFixed(3);
}
