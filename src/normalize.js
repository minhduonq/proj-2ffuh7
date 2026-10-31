__GENERATED_SOURCE__
;

export default function generate(THREE) {
  const pivot = new THREE.Group();
  pivot.add(__buildAsset(THREE));
  const root = new THREE.Group();
  root.add(pivot);
  root.updateMatrixWorld(true);
__TILT__
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);
  if (extent > 0 && Number.isFinite(extent)) {
    const center = box.getCenter(new THREE.Vector3());
    const scale = 0.99 / extent;
    root.scale.setScalar(scale);
    root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
  }
  return root;
}
