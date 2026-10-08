# Official starter kit pinned to the commit this work was checked against.
OFFICIAL_COMMIT := f980ef5059360fcf977ed7971af5c00ee5c1962f
PY := .venv/bin/python
INST := unicef world_bank unhcr

setup:            ## venv + official repo + practice data
	test -d raw/repo || git clone https://github.com/SituatedEvals/public.git raw/repo
	cd raw/repo && git checkout -q $(OFFICIAL_COMMIT)
	test -d .venv || python3 -m venv .venv
	.venv/bin/pip install -q numpy pandas pyarrow scipy scikit-learn pyyaml
	.venv/bin/pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
	$(MAKE) data

data:             ## official sandbox + harder sim2, all instruments
	for s in $(INST); do \
	  cd raw/repo && ../../$(PY) make_sandbox.py --schema data/$$s.json --out ../../data/sandbox/$$s >/dev/null; cd ../..; \
	  $(PY) tools/sim2.py --schema raw/repo/data/$$s.json --out data/sim2/$$s --seed 1; \
	done

eval:             ## make eval SUB=sub/v2 [DATA=data/sim2]
	$(PY) tools/evaluate.py $(SUB) --data $(or $(DATA),data/sim2)

zip:              ## make zip SUB=sub/v2 -> sub/v2.zip, then run the official checker
	rm -f $(SUB).zip && cd $(SUB) && zip -qr ../$(notdir $(SUB)).zip .
	cd raw/repo && ../../$(PY) tools/check_submission_zip.py ../../$(SUB).zip

official:         ## official score.py on the sandbox, all instruments (slow, subprocess)
	cd raw/repo && for s in $(INST); do ../../$(PY) score.py --submission ../../$(SUB) --data ../../data/sandbox/$$s --schema data/$$s.json --phase 1 | grep -E 'PASS|FAIL'; done
