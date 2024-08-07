# Style Bench

Release for the style control project for uniform Benchmarking of style control methods.

## Dependencies

- Python 3.10
- Black
- isort

## TODO

- [ ] Dataset
  - [x] Collect and format papers images
  - [ ] Entreprise grade styling images
    - [x] Collect and format
    - [ ] License : Check with Caroline

- [x] Implement Base Classes
  - [x] Metrics
  - [x] Models
  - [x] Datasets

- [ ] Implement Metrics
  - [x] CLIP Text
  - [x] CLIP Image
  - [ ] Dino (v1&2)
    - [ ] DinoV1
    - [x] DinoV2
  - [ ] Image Reward

- [ ] Implement Models
  - [x] Visualstyle
  - [x] IP-Adapter (& Instant Style) -> Diffusers
  - [x] StyleAligned

- [ ] Scripts
  - [ ] Inference
    - [ ] Add test for inference
      - [ ] StyleAlgined
      - [ ] VisualStyle
      - [ ] IP-Adapter (and InstantStyle)
  - [ ] Clean-up Examples scripts


- [ ] Dataset handling
  - [x] Webdataset format
  - [ ] Check for parquet dataset
  
## Tests

To run the tests, install pytest before (not in the requirements)
