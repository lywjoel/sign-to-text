# Calibration photos

`letters.html` runs MediaPipe's hand model on these photos and shows what the built-in fingerspelling rules
read from each one. It is how the rule thresholds in `src/fingerspell.ts` were checked against real
landmarks rather than only synthetic hands. Start `npm run dev`, then open
<http://localhost:5173/dev/letters.html>.

The photos in `img/` are letters of the Quebec Sign Language (LSQ) manual alphabet. For the letters
included here they use the same handshapes as SgSL. They were taken by Danielle-Claude Bélanger and released
into the public domain on Wikimedia Commons (for example
[File:LSQ e.jpg](https://commons.wikimedia.org/wiki/File:LSQ_e.jpg)).

There is one photo per letter from one signer, so treat a result here as a sanity check, not an accuracy
figure. M and N are expected to read as nothing: the thumb is hidden under the fingers, and the tracker's
guess for its position doesn't tell them apart.
