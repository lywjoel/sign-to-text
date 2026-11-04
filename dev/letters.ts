// Calibration page for the built-in fingerspelling rules (not part of the app).
// Runs MediaPipe's hand model on still photos of fingerspelled letters and
// shows what the rules read, with the measurements the rules use.
// Open http://localhost:5173/dev/letters.html while `npm run dev` is running.
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { explainLetter, handGeometry, pickLetter, scoreLetters } from '../src/fingerspell';
import { chiralityEvidence } from '../src/handshape';

/** Photos in dev/img, named LSQ_<letter>.jpg. M and N are expected to fail (hidden thumb). */
const LETTERS = 'abcdefghiklmny'.split('');

const out = document.getElementById('out')!;
const lines: string[] = [];
const log = (s: string) => {
  lines.push(s);
  out.textContent = lines.join('\n');
};
const f2 = (v: number) => v.toFixed(2);

const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
const landmarker = await HandLandmarker.createFromOptions(vision, {
  baseOptions: { modelAssetPath: '/models/hand_landmarker.task', delegate: 'CPU' },
  runningMode: 'IMAGE',
  numHands: 1,
  minHandDetectionConfidence: 0.3,
});

let correct = 0;
for (const letter of LETTERS.map((l) => l.toUpperCase())) {
  const img = new Image();
  img.src = `/dev/img/LSQ_${letter.toLowerCase()}.jpg`;
  try {
    await img.decode();
  } catch {
    log(`${letter}: photo missing`);
    continue;
  }
  // The photos are small close-ups; padding them gives the palm detector a typical hand size.
  const canvas = document.createElement('canvas');
  canvas.width = img.width * 2;
  canvas.height = img.height * 2;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#ccc';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, img.width / 2, img.height / 2);

  const result = landmarker.detect(canvas);
  if (!result.landmarks[0]) {
    log(`${letter}: no hand found`);
    continue;
  }
  const points = result.worldLandmarks[0].map((p) => ({ x: p.x, y: p.y, z: p.z }));
  // The photos show right hands.
  const g = handGeometry(points, 'right');
  const scores = scoreLetters(g);
  const picked = pickLetter(scores);
  if (picked?.letter === letter) correct++;

  const d = g.desc;
  const at = (i: number) => `${f2(d.local[i].x)},${f2(d.local[i].y)},${f2(d.local[i].z)}`;
  const conds = explainLetter(letter, g);
  log(
    [
      `${letter} read as ${picked?.letter ?? 'nothing'}${picked?.letter === letter ? '' : '  <--'}`,
      `  top ${scores.slice(0, 3).map((s) => `${s.letter} ${f2(s.score)}`).join(', ')}`,
      `  own conditions ${conds.shape.map(f2).join(' ')} / ${conds.orient.map(f2).join(' ')}`,
      `  reach ${d.reach.map(f2).join(' ')}  thumb tip ${at(4)}  index pip ${at(6)}  middle pip ${at(10)}  index tip ${at(8)}`,
      `  thumb to tips ${d.thumbTo.map(f2).join(' ')}  handedness label ${result.handedness[0]?.[0]?.categoryName}  shape says ${f2(chiralityEvidence(points))}`,
    ].join('\n'),
  );
}
log(`\n${correct}/${LETTERS.length} read correctly`);
document.title = 'done';
