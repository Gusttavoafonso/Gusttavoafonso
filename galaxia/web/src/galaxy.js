// Cena 3D: estrelas (documentos), nebulosas e linhas (constelações), marcador da pergunta.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const STAR_VERT = /* glsl */ `
  uniform float uTime, uDay, uPR, uDim, uFlash, uIntro;
  attribute vec3 aColor;
  attribute float aSize, aDay, aPhase, aHi;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float dist = max(-mv.z, 0.5);
    float born = step(aDay, uDay);
    float flash = uFlash * exp(-(uDay - aDay) / 50.0) * born;
    float twinkle = 0.78 + 0.22 * sin(uTime * (1.0 + aPhase * 2.0) + aPhase * 40.0);
    float keep = mix(1.0, mix(0.16, 1.0, aHi), uDim);
    float px = aSize * 46.0 * pow(dist, -0.62) * uPR;
    px *= 1.0 + flash * 1.6 + aHi * uDim * 0.9;
    // a entrada: as estrelas acendem do centro para a borda
    float intro = smoothstep(0.0, 0.25, uIntro * 1.25 - length(position) / 140.0 * 0.9 - aPhase * 0.1);
    vAlpha = born * twinkle * keep * intro * min(1.0, px / 2.0) * smoothstep(0.5, 4.0, dist);
    vColor = mix(aColor, vec3(1.0), flash * 0.7);
    gl_PointSize = clamp(px, 2.0, 80.0 * uPR);
    gl_Position = projectionMatrix * mv;
  }`;

const STAR_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    float glow = pow(1.0 - d, 2.4);
    float core = pow(1.0 - d, 9.0);
    gl_FragColor = vec4(mix(vColor, vec3(1.0), core * 0.85) * (0.55 + core), glow * vAlpha);
  }`;

const LINE_VERT = /* glsl */ `
  uniform float uDay;
  attribute vec3 aColor;
  attribute float aDay, aCluster;
  varying vec3 vColor;
  varying float vShow, vCluster;
  void main() {
    vColor = aColor; vShow = step(aDay, uDay); vCluster = aCluster;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const LINE_FRAG = /* glsl */ `
  uniform float uOpacity, uFocus;
  varying vec3 vColor;
  varying float vShow, vCluster;
  void main() {
    float f = uFocus < -0.5 ? 1.0 : (abs(vCluster - uFocus) < 0.5 ? 2.2 : 0.25);
    gl_FragColor = vec4(vColor, uOpacity * vShow * f);
  }`;

function radialTexture(stops, size = 128) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [o, col] of stops) grad.addColorStop(o, col);
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function ringTexture(size = 256) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  g.strokeStyle = '#fff';
  g.lineWidth = size * 0.018;
  g.beginPath();
  g.arc(size / 2, size / 2, size * 0.46, 0, Math.PI * 2);
  g.stroke();
  return new THREE.CanvasTexture(c);
}

const hash = (i) => {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export class Galaxy {
  constructor(canvas, labelRoot, data) {
    this.data = data;
    this.n = data.n;
    this.canvas = canvas;
    this.labelRoot = labelRoot;
    this.clusters = data.meta.clusters;
    this.day = data.meta.days;
    this.focus = -1;
    this.dimTarget = 0;
    this.introStart = null;
    this.pointer = { x: -1, y: -1, moved: false };
    this.onHover = () => {};
    this.onLabel = () => {};
    this.frameHooks = [];
    this.shift = 0;          // px: desloca o centro da cena quando o painel lateral abre
    this.shiftTarget = 0;
    this.shiftY = 0;         // idem, na vertical (painel inferior no celular)
    this.shiftYTarget = 0;
    this.reserved = () => [];   // retângulos de tela onde rótulos não devem aparecer

    const renderer = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' }));
    this.pr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(this.pr);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x03060f);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.5, 4000);
    this.camera.position.set(0, 120, 640);

    const controls = (this.controls = new OrbitControls(this.camera, canvas));
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.9;
    controls.minDistance = 4;
    controls.maxDistance = 900;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.28;
    this.lastInteraction = 0;
    controls.addEventListener('start', () => {
      controls.autoRotate = false;
      this.lastInteraction = performance.now();
      this.cancelFly?.();
    });

    this.#buildBackdrop();
    this.#buildStars();
    this.#buildNebulae();
    this.#buildLines();
    this.#buildLabels();
    this.#buildMarkers();

    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.85, 0.75, 0.12);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.resize();
    window.addEventListener('resize', () => this.resize());
    canvas.addEventListener('pointermove', (e) => {
      this.pointer.x = e.clientX;
      this.pointer.y = e.clientY;
      this.pointer.moved = true;
    });
    canvas.addEventListener('pointerleave', () => {
      this.pointer.x = -1;
      this.pointer.moved = true;
    });

    this.t0 = performance.now();
    renderer.setAnimationLoop(() => this.#frame());
  }

  // ---------------------------------------------------------------- construção
  #buildBackdrop() {
    const n = 2600;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const u = hash(i) * 2 - 1, th = hash(i + 0.5) * Math.PI * 2, r = 900 + hash(i + 0.25) * 900;
      const s = Math.sqrt(1 - u * u);
      pos.set([r * s * Math.cos(th), r * u, r * s * Math.sin(th)], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.scene.add(new THREE.Points(g, new THREE.PointsMaterial({
      color: 0x8ea2dc, size: 1.5, sizeAttenuation: false, transparent: true, opacity: 0.55, depthWrite: false,
    })));
    // névoa distante: dá profundidade ao fundo, como a faixa da Via Láctea
    const haze = radialTexture([[0, 'rgba(255,255,255,.5)'], [0.5, 'rgba(255,255,255,.12)'], [1, 'rgba(255,255,255,0)']]);
    for (const [x, y, z, size, color, opacity] of [
      [-900, 200, -1100, 2200, 0x2a4bd0, 0.16], [1100, -300, -900, 1900, 0x6a2fb8, 0.11],
      [200, 500, 1300, 2000, 0x1d5fa8, 0.10], [-1200, -400, 700, 1700, 0x8a5a20, 0.06],
    ]) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: haze, color, opacity, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
      s.position.set(x, y, z);
      s.scale.setScalar(size);
      this.scene.add(s);
    }
  }

  #buildStars() {
    const { n } = this;
    const { pos, cluster, day } = this.data;
    const color = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const phase = new Float32Array(n);
    const dayF = new Float32Array(n);
    this.hi = new Float32Array(n);
    const palette = this.clusters.map((c) => new THREE.Color(c.color));
    const dust = new THREE.Color('#56648f');
    const col = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const c = cluster[i];
      const h = hash(i);
      col.copy(c >= 0 ? palette[c] : dust);
      const v = c >= 0 ? 0.75 + hash(i + 0.3) * 0.5 : 0.7;
      color.set([col.r * v, col.g * v, col.b * v], i * 3);
      size[i] = (c >= 0 ? 1.0 : 0.7) * (0.85 + Math.pow(h, 7) * 3.2);
      phase[i] = hash(i + 0.7);
      dayF[i] = day[i];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(color, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.setAttribute('aDay', new THREE.BufferAttribute(dayF, 1));
    g.setAttribute('aPhase', new THREE.BufferAttribute(phase, 1));
    g.setAttribute('aHi', new THREE.BufferAttribute(this.hi, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 400);
    this.starUniforms = {
      uTime: { value: 0 }, uDay: { value: this.day }, uPR: { value: this.pr },
      uDim: { value: 0 }, uFlash: { value: 0 }, uIntro: { value: 0 },
    };
    this.stars = new THREE.Points(g, new THREE.ShaderMaterial({
      vertexShader: STAR_VERT, fragmentShader: STAR_FRAG, uniforms: this.starUniforms,
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    }));
    this.stars.frustumCulled = false;
    this.scene.add(this.stars);
  }

  #buildNebulae() {
    const tex = radialTexture([[0, 'rgba(255,255,255,.55)'], [0.35, 'rgba(255,255,255,.16)'], [1, 'rgba(255,255,255,0)']]);
    const { pos, cluster } = this.data;
    const sum = this.clusters.map(() => 0);
    for (let i = 0; i < this.n; i++) {
      const c = cluster[i];
      if (c < 0) continue;
      const [x, y, z] = this.clusters[c].centroid;
      sum[c] += Math.hypot(pos[i * 3] - x, pos[i * 3 + 1] - y, pos[i * 3 + 2] - z);
    }
    this.nebulae = this.clusters.map((c, k) => {
      c.radius = sum[k] / c.size;
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, color: c.color, transparent: true, opacity: 0, depthWrite: false, depthTest: false,
        blending: THREE.AdditiveBlending,
      }));
      s.position.fromArray(c.centroid);
      s.scale.setScalar(c.radius * 5.2);
      this.scene.add(s);
      return s;
    });
    // dias de nascimento ordenados, para saber quanto de cada constelação já existe
    const days = this.clusters.map(() => []);
    for (let i = 0; i < this.n; i++) if (cluster[i] >= 0) days[cluster[i]].push(this.data.day[i]);
    this.clusterDays = days.map((d) => Uint16Array.from(d).sort());
    this.born = this.clusters.map(() => 1);
  }

  #buildLines() {
    const { pos, day } = this.data;
    const P = [], C = [], D = [], K = [];
    const col = new THREE.Color();
    for (const c of this.clusters) {
      col.set(c.color);
      for (const [a, b] of c.edges) {
        const i = c.stars[a], j = c.stars[b];
        const d = Math.max(day[i], day[j]);
        for (const s of [i, j]) {
          P.push(pos[s * 3], pos[s * 3 + 1], pos[s * 3 + 2]);
          C.push(col.r, col.g, col.b);
          D.push(d);
          K.push(c.id);
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
    g.setAttribute('aColor', new THREE.Float32BufferAttribute(C, 3));
    g.setAttribute('aDay', new THREE.Float32BufferAttribute(D, 1));
    g.setAttribute('aCluster', new THREE.Float32BufferAttribute(K, 1));
    this.lineUniforms = { uDay: this.starUniforms.uDay, uOpacity: { value: 0.22 }, uFocus: { value: -1 } };
    this.lines = new THREE.LineSegments(g, new THREE.ShaderMaterial({
      vertexShader: LINE_VERT, fragmentShader: LINE_FRAG, uniforms: this.lineUniforms,
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    }));
    this.lines.frustumCulled = false;
    this.scene.add(this.lines);
  }

  #buildLabels() {
    this.labels = this.clusters.map((c) => {
      const el = document.createElement('button');
      el.className = 'label';
      el.textContent = c.name;
      el.style.setProperty('--c', c.color);
      el.style.opacity = 0;
      el.addEventListener('click', () => this.onLabel(c.id));
      this.labelRoot.appendChild(el);
      return { el, w: 0, v: new THREE.Vector3().fromArray(c.centroid), shown: false };
    });
  }

  #buildMarkers() {
    const glow = radialTexture([[0, 'rgba(255,250,235,1)'], [0.12, 'rgba(245,214,138,.9)'], [0.4, 'rgba(245,214,138,.18)'], [1, 'rgba(245,214,138,0)']]);
    const ring = ringTexture();
    const sprite = (map, color) => new THREE.Sprite(new THREE.SpriteMaterial({
      map, color, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    }));
    this.query = new THREE.Group();
    this.query.visible = false;
    this.queryGlow = sprite(glow, 0xffffff);
    this.queryRings = [sprite(ring, 0xf5d68a), sprite(ring, 0xf5d68a)];
    this.query.add(this.queryGlow, ...this.queryRings);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(64 * 6), 3));
    this.queryLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({
      color: 0xf5d68a, transparent: true, opacity: 0.55, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    }));
    this.queryLines.frustumCulled = false;
    this.queryLines.visible = false;
    this.scene.add(this.query, this.queryLines);

    this.selection = sprite(ring, 0x8fd8ff);
    this.selection.visible = false;
    this.scene.add(this.selection);
  }

  // ---------------------------------------------------------------- API
  starPosition(i, out = new THREE.Vector3()) {
    return out.fromArray(this.data.pos, i * 3);
  }

  start() {
    this.introStart = performance.now();
    this.flyTo(new THREE.Vector3(0, 0, 0), 215, { duration: 4200, dir: new THREE.Vector3(0.25, 0.42, 1), keepRotate: true });
  }

  setDay(day, flash = 0) {
    this.day = day;
    this.starUniforms.uDay.value = day;
    this.starUniforms.uFlash.value = flash;
    this.clusterDays.forEach((d, k) => {
      let lo = 0, hi = d.length;
      while (lo < hi) { const m = (lo + hi) >> 1; if (d[m] <= day) lo = m + 1; else hi = m; }
      this.born[k] = lo / d.length;
    });
  }

  /** Destaca um conjunto de estrelas (ou limpa, com null). */
  highlight(indices) {
    this.hi.fill(0);
    if (indices) for (const i of indices) this.hi[i] = 1;
    this.stars.geometry.attributes.aHi.needsUpdate = true;
    this.dimTarget = indices ? 1 : 0;
  }

  focusCluster(id) {
    this.focus = id ?? -1;
    this.lineUniforms.uFocus.value = this.focus;
    if (this.focus < 0) return this.highlight(null);
    const { cluster } = this.data;
    this.hi.fill(0);
    for (let i = 0; i < this.n; i++) if (cluster[i] === this.focus) this.hi[i] = 1;
    this.stars.geometry.attributes.aHi.needsUpdate = true;
    this.dimTarget = 0.85;
    const c = this.clusters[this.focus];
    return this.flyTo(new THREE.Vector3().fromArray(c.centroid), Math.max(26, c.radius * 3.4));
  }

  showQuery(position, neighbors) {
    this.focus = -1;
    this.lineUniforms.uFocus.value = -1;
    this.query.position.copy(position);
    this.query.visible = true;
    this.queryT0 = performance.now();
    this.queryNeighbors = neighbors.slice(0, 64);
    this.queryLines.visible = true;
    this.highlight(neighbors);
  }

  clearQuery() {
    this.query.visible = false;
    this.queryLines.visible = false;
    this.queryNeighbors = null;
  }

  select(i) {
    this.selected = i ?? -1;
    this.selection.visible = this.selected >= 0;
    if (this.selected >= 0) this.starPosition(this.selected, this.selection.position);
  }

  /** Voa até `target`, parando a `distance` dele. */
  flyTo(target, distance, { duration = 2300, dir = null, keepRotate = false } = {}) {
    this.cancelFly?.();
    const cam = this.camera, ctl = this.controls;
    const p0 = cam.position.clone(), t0 = ctl.target.clone();
    // chega "por fora": a câmera fica entre o alvo e o espaço, com a galáxia ao fundo
    const d = dir ? dir.clone().normalize() : target.clone().add(new THREE.Vector3(0.001, 0.35 * Math.max(target.length(), 20), 0)).normalize();
    if (!dir && target.length() < 8) d.copy(p0).sub(target).normalize();
    const p1 = target.clone().addScaledVector(d, distance);
    const start = performance.now();
    if (!keepRotate) ctl.autoRotate = false;
    this.lastInteraction = start;
    return new Promise((resolve) => {
      let live = true;
      this.cancelFly = () => { live = false; this.cancelFly = null; resolve(false); };
      const step = () => {
        if (!live) return;
        const t = Math.min(1, (performance.now() - start) / duration), e = ease(t);
        cam.position.lerpVectors(p0, p1, e);
        ctl.target.lerpVectors(t0, target, Math.min(1, e * 1.25));
        if (t < 1) requestAnimationFrame(step);
        else { this.cancelFly = null; resolve(true); }
      };
      step();
    });
  }

  resetView() {
    this.clearQuery();
    this.select(null);
    this.focusCluster(null);
    this.controls.autoRotate = true;
    return this.flyTo(new THREE.Vector3(0, 0, 0), 215, { dir: new THREE.Vector3(0.25, 0.42, 1), keepRotate: true });
  }

  resize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.composer.setPixelRatio(this.pr);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    if (this.shift || this.shiftY) this.camera.setViewOffset(w, h, this.shift, this.shiftY, w, h);
    this.camera.updateProjectionMatrix();
    for (const l of this.labels) l.w = 0;
  }

  // ---------------------------------------------------------------- por quadro
  #pick() {
    const { x, y } = this.pointer;
    if (x < 0) return -1;
    const m = new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse).elements;
    const { pos, day } = this.data, hi = this.hi, dimmed = this.starUniforms.uDim.value > 0.5;
    const hw = this.w / 2, hh = this.h / 2;
    let best = -1, bestScore = 14 * 14;
    for (let i = 0, j = 0; i < this.n; i++, j += 3) {
      if (day[i] > this.day || (dimmed && !hi[i])) continue;
      const px = pos[j], py = pos[j + 1], pz = pos[j + 2];
      const w = m[3] * px + m[7] * py + m[11] * pz + m[15];
      if (w < 1) continue;
      const sx = (m[0] * px + m[4] * py + m[8] * pz + m[12]) / w * hw + hw - x;   // a matriz de projeção já inclui o deslocamento
      if (sx > 14 || sx < -14) continue;
      const sy = hh - (m[1] * px + m[5] * py + m[9] * pz + m[13]) / w * hh - y;
      const score = sx * sx + sy * sy + w * 0.25;   // empate: prefere a mais próxima da câmera
      if (score < bestScore) { bestScore = score; best = i; }
    }
    return best;
  }

  #updateLabels() {
    const v = new THREE.Vector3();
    const placed = this.reserved();
    const camPos = this.camera.position;
    for (let k = 0; k < this.labels.length; k++) {
      const l = this.labels[k], c = this.clusters[k];
      if (!l.w) l.w = l.el.offsetWidth || c.name.length * 9 + 30;
      v.copy(l.v).project(this.camera);
      const dist = camPos.distanceTo(l.v);
      let o = 0;
      if (v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05 && this.born[k] > 0.12) {
        const x = (v.x * 0.5 + 0.5) * this.w, y = (-v.y * 0.5 + 0.5) * this.h - 6;   // a projeção já inclui o deslocamento
        const box = [x - 4, y - 14, x + l.w + 4, y + 14];
        const focused = this.focus === k;
        const clash = !focused && placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1]);
        if (!clash) {
          placed.push(box);
          o = Math.min(1, (this.born[k] - 0.12) * 5) * THREE.MathUtils.smoothstep(dist, 6, 22) * (1 - THREE.MathUtils.smoothstep(dist, 520, 760));
          if (this.focus >= 0 && !focused) o *= 0.3;
          if (this.query.visible) o *= 0.6;
          l.el.style.transform = `translate3d(${x.toFixed(1)}px,${(y - 14).toFixed(1)}px,0)`;
        }
      }
      o *= this.introAlpha ?? 0;
      if ((o > 0.02) !== l.shown) { l.shown = o > 0.02; l.el.style.pointerEvents = l.shown ? 'auto' : 'none'; }
      l.el.style.opacity = o.toFixed(2);
    }
  }

  #frame() {
    const now = performance.now();
    const t = (now - this.t0) / 1000;
    const u = this.starUniforms;
    u.uTime.value = t;
    u.uDim.value += (this.dimTarget - u.uDim.value) * 0.08;
    const intro = this.introStart ? Math.min(1, (now - this.introStart) / 3800) : 0;
    u.uIntro.value = intro;
    this.introAlpha = THREE.MathUtils.smoothstep(intro, 0.75, 1);

    if (!this.controls.autoRotate && !this.cancelFly && this.idleRotate && now - this.lastInteraction > 30000) this.controls.autoRotate = true;
    this.controls.update();
    if (Math.abs(this.shiftTarget - this.shift) > 0.5 || Math.abs(this.shiftYTarget - this.shiftY) > 0.5) {
      this.shift += (this.shiftTarget - this.shift) * 0.07;
      this.shiftY += (this.shiftYTarget - this.shiftY) * 0.07;
      this.camera.setViewOffset(this.w, this.h, this.shift, this.shiftY, this.w, this.h);
    }

    const dimK = 1 - u.uDim.value * 0.75;
    this.nebulae.forEach((s, k) => {
      const f = this.focus === k ? 1.6 : 1;
      // some quando a câmera entra na nebulosa, para não tingir a tela inteira
      const r = this.clusters[k].radius;
      const near = THREE.MathUtils.smoothstep(this.camera.position.distanceTo(s.position), r * 1.2, r * 5);
      s.material.opacity = 0.085 * this.born[k] * this.introAlpha * (this.focus === k ? 1 : dimK) * f * near;
    });
    this.lineUniforms.uOpacity.value = 0.24 * this.introAlpha * (this.query.visible ? 0.35 : 1);

    if (this.query.visible) {
      const age = (now - this.queryT0) / 1000;
      const d = this.camera.position.distanceTo(this.query.position);
      const base = THREE.MathUtils.clamp(d * 0.075, 1.2, 16);
      this.queryGlow.scale.setScalar(base * (1.5 + 0.2 * Math.sin(t * 3)) * Math.min(1, age * 2.5));
      this.queryRings.forEach((r, k) => {
        const ph = (age * 0.55 + k * 0.5) % 1;
        r.scale.setScalar(base * (0.5 + ph * 2.6));
        r.material.opacity = (1 - ph) * 0.7;
      });
      // as linhas "crescem" da pergunta até cada vizinha
      const arr = this.queryLines.geometry.attributes.position.array;
      const q = this.query.position, p = this.data.pos;
      this.queryNeighbors.forEach((i, k) => {
        const g = ease(THREE.MathUtils.clamp(age * 1.1 - 0.5 - k * 0.07, 0, 1));
        arr.set([q.x, q.y, q.z, q.x + (p[i * 3] - q.x) * g, q.y + (p[i * 3 + 1] - q.y) * g, q.z + (p[i * 3 + 2] - q.z) * g], k * 6);
      });
      this.queryLines.geometry.setDrawRange(0, this.queryNeighbors.length * 2);
      this.queryLines.geometry.attributes.position.needsUpdate = true;
    }
    if (this.selection.visible) {
      const d = this.camera.position.distanceTo(this.selection.position);
      this.selection.scale.setScalar(THREE.MathUtils.clamp(d * 0.045, 0.5, 9) * (1 + 0.08 * Math.sin(t * 4)));
    }

    if (this.pointer.moved) {
      this.pointer.moved = false;
      const i = this.cancelFly ? -1 : this.#pick();
      if (i !== this.hovered) { this.hovered = i; this.canvas.style.cursor = i >= 0 ? 'pointer' : ''; }
      this.onHover(i, this.pointer.x, this.pointer.y);
    }
    this.#updateLabels();
    for (const f of this.frameHooks) f(now);
    this.composer.render();
  }
}
