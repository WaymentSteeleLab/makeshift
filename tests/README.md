# Tests

Run all tests:

```bash
python -m unittest discover tests/ -v
```

Or run individually:

```bash
python -m unittest tests/test_reref_lacs.py -v
python -m unittest tests/test_reref_panav.py -v
```

## Notes

Both tests fetch BMRB entry 5363 and apply the same GLY HA averaging used in
`demo_lacs.ipynb` and `demo_panav.ipynb`. Expected offset values are taken
directly from those notebooks and asserted to 6 decimal places.

No external dependencies beyond the standard library — no pytest required.
