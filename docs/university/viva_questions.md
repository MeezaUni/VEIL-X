---
title: "VEIL-X Viva Preparation"
subtitle: One-page revision sheet and 35 questions with answers
---

# One-page explanation (read this first)

**What it is.** VEIL-X is a local AI application that compares two face photos and says MATCH,
NON-MATCH or ABSTAIN ("not sure"). It runs on a laptop, with no internet, and explains every
result.

**How it works, in order:**
1. **Check the files.** Format, size, corruption.
2. **Find and align the face.** The SCRFD detector finds the face and five landmarks. The face
   is rotated and scaled to a standard 112×112 crop. If there are several faces, the user must
   pick one.
3. **Turn each face into numbers.** A pretrained deep network, AdaFace IR-101, turns each face
   into a 512-number *embedding*. The length of the embedding before normalisation, the *feature
   norm*, works as a quality score: blurry or covered faces give smaller norms.
4. **Compare.** Cosine similarity *s* between the two embeddings, and the pair quality *q*, the
   smaller norm.
5. **Score.** A logistic-regression formula combines *s*, *q* and *s×q* into a score between 0
   and 1.
6. **Decide.** The pair is put into a low-, medium- or high-quality group. Each group has its own
   thresholds, chosen on held-out calibration data.
   - Low-quality pairs whose score falls in the middle band give **ABSTAIN**.
   - If *q* is below anything seen during calibration, the app also abstains (the *support
     guard*).
7. **Explain.** The app shows the numbers, the threshold bar, the reason for abstaining, and
   heat maps of predicted face visibility. The heat maps are explanation only.

**What was trained and what was not.**
- The face recognisers (AdaFace, ArcFace) and the detector (SCRFD) are *pretrained* and were not
  changed.
- This project trained only:
  - a small visibility network (U-Net style);
  - the small logistic formula;
  - the thresholds.

**Main results:**
- The app reproduces the research code exactly (1000/1000 decisions).
- CPU and GPU agree.
- About 0.23 s per photo pair on an RTX 4060 laptop.
- More than 400 automated tests pass.

**Main lessons (be ready to explain these):**
1. Learned visibility looked promising but failed on new occluder types, so quality, not
   visibility, drives decisions.
2. The 5 % error target held on data like the calibration data, but not on a harder group of
   identities, as confirmed twice. Thresholds depend on the population.
3. Very low-quality inputs (even random noise) could get a false MATCH from the research formula,
   so the app abstains there.

**What it cannot claim.**
- It is not certified, not production-ready and not tested on real physically occluded photos.
- A score is not a probability of identity.

# Questions and answers

## A. The project

**1. What does VEIL-X do?**
It compares two face images on your own computer and answers MATCH, NON-MATCH or ABSTAIN, with
explanations: similarity, quality, the threshold region, the abstention reason and visibility
heat maps.

**2. Why is face verification useful?**
It confirms that a person is who they claim to be, for example when unlocking a phone or at a
passport gate, by comparing a live photo with a reference photo.

**3. What is the difference between verification and identification?**
- *Verification* (1:1): are these two images the same person?
- *Identification* (1:N): who is this, out of a database of many people?

VEIL-X only does verification. It has no database and does no searching, by design (privacy).

**4. Why add ABSTAIN?**
Forcing an answer on a blurred or heavily covered face produces silent errors. Abstaining hands
the case to a human or asks for a better photo. It trades coverage (how many pairs get a
decision) for fewer wrong decisions.

**5. What makes VEIL-X different from a basic face matcher?**
- three outcomes instead of two;
- quality-dependent thresholds;
- explanations and heat maps;
- strict local privacy;
- a documented test and limitations record.

It does **not** use a new recognition network.

## B. Deep learning basics

**6. What is a CNN?**
A convolutional neural network applies learned filters across the image to detect patterns
(edges, then parts, then whole faces), layer by layer.

**7. What is a face embedding?**
A fixed-length vector, here 512 numbers, produced by the network. It is trained so that the same
person's images give vectors pointing in similar directions, and different people's give
different directions.

**8. What is AdaFace?**
A face-recognition training method (Kim et al., CVPR 2022). It adapts the training margin to
image quality, using the feature norm as a quality proxy. We use a pretrained AdaFace IR-101
model; IR-101 is a 101-layer residual network.

**9. What is ArcFace and how is it used here?**
ArcFace (Deng et al., CVPR 2019) is an earlier, widely used margin-based method. In VEIL-X its
similarity is shown **for comparison only**; it is not used in the decision.

**10. What is SCRFD?**
A fast face detector (Guo et al., ICLR 2022). It outputs face boxes and five landmarks (eyes,
nose, mouth corners), which we use to align the face.

**11. Why align faces?**
The recognition model expects faces in a standard position and size (112×112, eyes at fixed
points). Alignment makes comparisons consistent.

**12. Did you train the face recogniser?**
No. AdaFace, ArcFace and SCRFD are pretrained and frozen. We trained a small visibility network,
fitted a small logistic formula, and calibrated thresholds.

## C. Similarity, quality and decisions

**13. What is cosine similarity?**
The cosine of the angle between two embeddings: 1 means the same direction, 0 unrelated. Higher
values mean more likely the same person.

**14. How does VEIL-X measure image quality?**
By the AdaFace feature norm, the length of the embedding before normalisation. Good, clear faces
give larger norms. The pair quality is the smaller of the two norms.

**15. What is logistic fusion?**
A logistic-regression formula combines similarity, quality and their product into one score
between 0 and 1. It was fitted on training pairs.

**16. What is calibration?**
Choosing the decision thresholds on separate data, so that the error rates on accepted decisions
meet a target, here 5 %, on data like the calibration data. We used Learn-then-Test, a
statistical method that tests candidate thresholds with exact binomial tests.

**17. What are a false match and a false non-match?**
- *False match*: two different people judged the same. This is the security risk.
- *False non-match*: the same person judged different. This is the convenience risk.

**18. What are selective error rates and coverage?**
Errors counted only among the pairs the system actually decided. Coverage is the fraction
decided. Abstaining more lowers the selective error but also lowers coverage.

**19. Why are there three quality groups?**
Low-quality pairs are harder, so they get stricter, separate thresholds, including an abstention
band. High-quality pairs can use a single threshold.

**20. Does a score of 1.000 mean 100 % certain?**
No. It is the output of a fitted formula, not a probability that two images show the same
person. Two noise images could get 1.000 from the research formula, which is why the app's
support guard exists.

## D. Visibility and occlusion

**21. What is a visibility map?**
A 28×28 grid predicting which parts of the face are visible (blue) or covered (orange), produced
by our small visibility network.

**22. What is U-Net?**
An encoder–decoder CNN with skip connections, originally for image segmentation (Ronneberger et
al., 2015). Our visibility network is a small U-Net-style model with 3.6 M parameters.

**23. What is IoU?**
Intersection over union: the overlap between the predicted covered area and the true covered
area, divided by their combined area. 1 is perfect, 0 is no overlap.

**24. Why is visibility not used in the decision?**
On unseen random patches the network called covered areas "visible" (IoU 0.13). A visibility-based
rule then had 8.52 % false non-matches, against 2.12 % for the quality-based rule. The project
kept visibility as an explanation only.

**25. What is co-visibility?**
How much face area is visible in both images. It did not consistently improve decisions beyond
quality, and it hurt on unseen occluders, so it is not used.

## E. Data, testing and limits

**26. What data did you use?**
- Synthetic DigiFace-1M faces, computer-generated, with no real people, for training,
  calibration and testing.
- Synthetic occlusions: masks, sunglasses, hands, random patches, lighting, and a test-only hair
  occluder.
- Some real LFW photographs with synthetic overlays, used under academic authorisation.

**27. What is identity-disjoint evaluation?**
Training, calibration and test sets contain different people. The system is never tested on
someone it was tuned on.

**28. What is a confidence interval and why use one?**
A range that likely contains the true error rate. We used identity-level (jackknife) intervals,
because pairs of the same person are not independent.

**29. Why did the 5 % target fail?**
The thresholds were calibrated mostly on one kind of synthetic identity. On a harder group
(identities with 72 images, more variation), false non-matches rose to about 5.2–6.5 %. A second,
independent test on 4,000 new identities confirmed this. This is *population shift*:
calibration only holds for data like the calibration data.

**30. What is the low-quality false-match problem, and what is the support guard?**
- *The problem.* Below the quality range seen in calibration, the formula extrapolates. On new
  test data, the false-match rate there was 22.97 %, against 1.50 % above it. Random noise even
  scored 1.000.
- *The support guard.* The app abstains whenever quality is below the calibrated range. It is an
  application safeguard, not a proven universal fix. The original research results are reported
  unchanged.

**31. How did you test the application?**
- More than 400 automated tests (CPU and GPU) plus strict code checks.
- Exact parity with the research pipeline: 1000/1000 decisions.
- CPU/GPU agreement: 100/100.
- A browser check of local-only networking.
- A privacy scan.
- Live checks of every refusal case.

**32. How does GPU acceleration help?**
The deep models run on the RTX 4060 GPU in half precision: about 18–25 ms per pair per model,
against about 140–165 ms on the CPU. Face detection still runs on the CPU, about 0.17 s.

## F. Privacy, limitations, future work

**33. How is privacy protected?**
- The app listens only on 127.0.0.1, and a browser security policy blocks third-party requests.
- There is no cloud, no stored embeddings, no database and no analytics.
- Uploads are deleted after reading, and leftover display files expire within about three
  minutes.

**34. What are the main limitations?**
- Synthetic test data only; no real physical occlusions evaluated.
- Calibration depends on the population.
- The low-quality issue, guarded only in the app.
- Five configuration-specific calibrations, none universal.
- Not certified, and not for real identity decisions.

**35. What would you do next?**
- Re-calibrate on the intended users' photos and camera.
- Test on real occluded photos (e.g. IJB-C).
- Use a fused score that cannot extrapolate.
- Run detection on the GPU.

# What you must personally be able to explain

- The pipeline in seven steps (see the one-page explanation above).
- Why ABSTAIN is useful, and what coverage means.
- Embedding and cosine similarity; feature norm as quality.
- Why visibility is shown but not used.
- Population shift, and why the 5 % target failed.
- The low-quality guard, and why a score of 1.000 is not certainty.
- What was pretrained versus what you trained or calibrated.
