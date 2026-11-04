# SgSL to text

Sign to your camera in Singapore Sign Language (SgSL) and read it back as text. Everything runs in the
browser, so the video never leaves your device.

## What it can and can't do

- **Fingerspelling works out of the box.** The SgSL one-handed alphabet is built in, including J and Z,
  which are read from their movement. M, N and T hide the thumb under the fingers, where tracking can only
  guess, so teach those three (see below) if you use them.
- **Word signs need examples from you.** There is no ready-made SgSL vocabulary model, so you record three to
  five examples of each sign and the app matches against them. A match takes into account:
  - both hands' handshapes and which way they face
  - where the hands are relative to your face, head and body, and what they touch
  - how the hands move and how they relate to each other
  - your facial expression, head movement and arm position
- **Output is SgSL gloss in signing order**, such as `YESTERDAY TABLE I BUY`. It is not translated into
  English.
- It recognises the signs you taught, works best for the person who recorded them, and needs a short pause
  between signs.

## Run it

You need Node.js 20 or newer.

```sh
npm install
npm run dev
```

Open <http://localhost:5173> and press **Start camera**. The first run downloads about 30 MB of tracking
models into `public/models/` (you can also do this with `npm run setup`). If a download fails, the app
loads the models from Google's servers instead.

To try it on a phone, run `npm run dev:phone` and open `https://<your computer's IP>:5173` on the phone. The
camera only works over HTTPS, so this uses a self-signed certificate; accept the browser's warning once.

`npm run build` produces a static site in `dist/` (about 57 MB, mostly models) that you can host on any
HTTPS server.

## Using it

1. Sit so that your head, shoulders and hands down to your waist are in view, in even light.
2. Choose what to recognise at the top: letters and signs, only fingerspelling, or only signs.
3. Watch the line under the video. It shows **Resting**, **Moving** or **Holding**. Pause briefly at the
   end of each letter and each sign so it reaches Holding.
4. Drop your hands below the dashed line to end a fingerspelled word. A long rest starts a new line.
5. Words that may still change are shown faded. For example, a letter is withdrawn if it turns out to be
   the start of a sign.

Keyboard: <kbd>Backspace</kbd> undoes the last word, <kbd>Space</kbd> ends a word, <kbd>Enter</kbd> starts a
new line.

### Teaching a sign

1. On the **Teach** tab, type the gloss (suggestions cover common SgSL signs) and choose **Word sign** or
   **Letter**.
2. Press **Record example**. During the 3-second countdown, put your hands where the sign starts. Sign
   once at your normal pace, then hold the last position still until recording stops.
3. Watch the replay, then keep or discard it. The review warns you if the example looks like a different
   sign you already taught.
4. Record three to five examples per sign. Examples from different days or lighting help.

The **Library** tab lists what you've taught. Use Export and Import to back up or share it as a JSON file.

### Settings worth knowing

- **Dominant hand:** set Left if you sign left-handed. Examples are stored relative to your dominant hand.
- **Matching:** stricter gives fewer wrong words but more missed ones.
- **Pause length:** shorten this if you sign quickly.
- **Tracking:** separate hand, body and face models (the default, uses the GPU) or one combined model.
  The combined model's face-expression part can't run on the browser's GPU backend, so it runs on the
  CPU and is slower.
- **Show tracking numbers:** displays the live motion measurements, useful when tuning the two sliders.

## How it works

1. **Tracking.** MediaPipe Tasks Vision finds 21 points on each hand, 33 on the body, the face mesh and
   12 facial-expression channels in every frame.
2. **Normalising** (`skeleton.ts`, `features.ts`).
   - Hands are labelled dominant and non-dominant; a left-handed signer's frames are mirrored.
   - Each frame becomes 118 numbers: handshape (joint angles, finger spread, thumb position), palm
     orientation, hand position in shoulder widths, and closeness to the forehead, eyes, nose, mouth, chin,
     cheeks, ears, chest and shoulder.
   - The rest covers the two hands relative to each other, facial expression, head tilt, turn and nod, and
     arm position.
3. **Segmenting** (`segmenter.ts`). Smoothed hand speed and handshape change split the stream into rests,
   movements and holds.
4. **Recognising** (`recognizer.ts`).
   - **Holds** are read as letters, using the built-in rules in `fingerspell.ts` or your taught letters,
     or as held signs. A change between near-identical letters in place, such as M to N or U to V, also
     counts.
   - **Movements** are compared with your examples using subsequence dynamic time warping (`dtw.ts`), which
     tolerates different speeds and extra movement before and after the sign.
   - Each sign's acceptance threshold adapts to how consistent its examples are.
   - Signs made of several movements with pauses in between are supported.
   - Raising the hands into position isn't mistaken for a sign.
5. **Transcript** (`transcriber.ts`). Letters are joined into words, signs are added as gloss, and
   provisional words are confirmed after a moment.

Other files: `tracker.ts` (MediaPipe setup and GPU/CPU fallback), `library.ts` (examples in IndexedDB),
`recorder.ts` (Teach recording and trimming), `overlay.ts` (drawing), `main.ts` (the page).

## Tests

`npm test` runs 115 tests:

- the hand geometry on a synthetic jointed hand
- every static letter, including tilted, noisy and left-handed versions
- the DTW matcher, the segmenter and the transcript
- end-to-end recognition of taught signs, including two-handed and multi-part signs
- a robustness check with tracking-like noise and natural variation in position, size and speed. Across
  four random seeds it got 29–30 of 30 taught signs right, gave no wrong words, and accepted 0–2 of 18
  untaught movements as signs.

The built-in letter rules were also checked against MediaPipe's landmarks for real photographs; see
`dev/README.md` (12 of 14 letters read correctly; M and N read as nothing).

Testing used synthetic data and still photos, not live signing. Thresholds may need adjusting for your
camera and signing, which is what the Matching and Pause length settings are for.

## Privacy

Video is processed in the browser and never uploaded. Taught examples are stored in this browser's
IndexedDB and only leave it if you export them.

## Credits

- Tracking: [MediaPipe](https://ai.google.dev/edge/mediapipe) models and runtime (Apache 2.0).
- Typeface: Atkinson Hyperlegible Next by the Braille Institute (SIL Open Font License).
- Alphabet and vocabulary reference: the [SgSL Sign Bank](https://blogs.ntu.edu.sg/sgslsignbank/) by NTU and
  the Singapore Association for the Deaf. The suggested sign names are labels only; no sign data is
  included.
- The interface colours follow Singapore's MRT line colours.
