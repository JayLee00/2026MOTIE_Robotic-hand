"""Evaluation subpackage (US-008): acceptance-criteria scoring for the tracker.

Computes and prints a pass/fail table over a recorded sequence:

  * AC-3 / AC-3b  : world surface-centroid-to-GT distance (diagnostic, <= 6 cm).
  * AC-5 / AC-10  : ellipsoid_fit center L2 (median/p90), (a, c) rel-err,
                    elongated major-axis angle; with an ANTI-REGRESSION guard
                    that the diagnostic surface centroid stays ~one radius off
                    the fitted center (the bias must NOT collapse to ~0).
  * AC-11         : 3D bbox IoU vs the GT enclosing box (1-axis best case +
                    3-axis achieved IoU-vs-center-error curves).
  * AC-12         : geometry-path throughput (deproject+roi+fit+bbox, EXCL seg)
                    >= 30 Hz median over >= 300 frames, plus SAM2-path Hz.
  * AC-2          : a grep gate that the package never imports the sim-side stack.

When ``gt_shape.json`` is absent (e.g. the soldering reference sequence), the
fruit-GT rows (AC-5 / AC-10 / AC-11) are marked ``N/A-no-fruit-GT`` while
AC-3 / AC-3b / AC-8 / AC-12 are still computed.

Pure numpy / cv2 / scipy / yaml. No sim-side imports.
"""

from .evaluate import evaluate_sequence, main

__all__ = ["evaluate_sequence", "main"]
