# run cloned code

# brain

```bash
python -m nii2dcm   ./test/nifti_las/reo-SVR-output-brain.nii.gz   ./test/dcm   -d SVR   -r ./test/source_las/MR000000.dcm
```

# body

```bash
python -m nii2dcm   ./test/nifti_ras/reo-DSVR-output-body.nii.gz   ./test/dcm   -d SVR   -r ./test/source_ras/MR000000.dcm
```