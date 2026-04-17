# Technical Appendix: PIV Processing Recipe

## Acquisition metadata to record
- Camera model, resolution, pixel pitch
- Lens focal length, aperture, working distance
- Frame rate, exposure time, dt
- Laser sheet thickness, pulse energy, timing jitter
- Seeding particle type and estimated particles-per-pixel
- Geometry, free-stream velocity, Reynolds number

## Processing pipeline (OpenPIV style)
1. Preprocess: grayscale, flat-field correction, high-pass filter.
2. Interrogation: multi-pass 64->32 px windows, 50% overlap, window deformation.
3. Validation: peak2peak SNR threshold (start 1.2), global median filter, local median replacement.
4. Postprocess: compute mean/RMS, vorticity, Q-criterion, phase-averaging if periodic.
5. Uncertainty: synthetic image tests, repeatability (N pairs), SNR sensitivity study.

## Deliverables per run
- Raw images (archived)
- Processing script and parameter file
- Vector fields (binary or CSV)
- Vorticity maps and figures
- One-page uncertainty note
