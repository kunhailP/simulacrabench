# Official starter kit pinned to the commit this work was checked against.
OFFICIAL_COMMIT := f980ef5059360fcf977ed7971af5c00ee5c1962f
PY := .venv/bin/python
INST := unicef world_bank unhcr

setup:            ## venv + official repo + practice data
	test -d raw/repo || git clone https://github.com/SituatedEvals/public.git raw/repo
	cd raw/repo && git checkout -q $(OFFICIAL_COMMIT)
	test -d .venv || python3 -m venv .venv
	.venv/bin/pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
	.venv/bin/pip install -q -r requirements-lab.txt
	$(MAKE) data

data:             ## official sandbox + harder sim2, all instruments
	for s in $(INST); do \
	  cd raw/repo && ../../$(PY) make_sandbox.py --schema data/$$s.json --out ../../data/sandbox/$$s >/dev/null; cd ../..; \
	  $(PY) tools/sim2.py --schema raw/repo/data/$$s.json --out data/sim2/$$s --seed 1; \
	done

WORLDS := base weakx strongx cells cleangate ordinal ordweak notrait

worlds:           ## oracle worlds (CPU, parallel, ~15 min for UNHCR): data/worlds/<scenario>/<inst>
	for sc in $(WORLDS); do for s in $(INST); do \
	  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $(PY) tools/world.py --schema raw/repo/data/$$s.json \
	    --scenario $$sc --out data/worlds/$$sc/$$s --reps 2000 --seed 7 & \
	done; done; wait

eval:             ## make eval SUB=sub/v2 [DATA=data/sim2]
	$(PY) tools/evaluate.py $(SUB) --data $(or $(DATA),data/sim2)

zip:              ## make zip SUB=sub/v2 -> sub/v2.zip, then run the official checker
	rm -f $(SUB).zip && cd $(SUB) && zip -qr ../$(notdir $(SUB)).zip . -x '__pycache__/*' '*.pyc'
	cd raw/repo && ../../$(PY) tools/check_submission_zip.py ../../$(SUB).zip

official:         ## official score.py on the sandbox, all instruments (slow, subprocess)
	cd raw/repo && for s in $(INST); do ../../$(PY) score.py --submission ../../$(SUB) --data ../../data/sandbox/$$s --schema data/$$s.json --phase 1 | grep -E 'PASS|FAIL'; done

lab-data:         ## real-data lab: GSS download (hash-checked) -> data/proxy/gss, data/proxy/gss_hcr
	tools/fetch_gss.sh data/raw/gss
	$(PY) tools/proxy_gss.py --src data/raw/gss --out data/proxy/gss
	$(PY) tools/proxy_gss_hcr.py --src data/proxy/gss --out data/proxy/gss_hcr

lab:              ## make lab SUB=sub/v8 [LAB=gss] [REPS=3]: repeated-split paired evaluation
	V8_BUDGET=100000 $(PY) tools/lab.py $(SUB) --data data/proxy/$(or $(LAB),gss) --reps $(or $(REPS),3)

compare:          ## make compare A=v7 B=v8 [LAB=gss]: paired difference of two lab runs
	$(PY) tools/lab.py --compare $(A) $(B) --data data/proxy/$(or $(LAB),gss)

tabicl:           ## fetch the TabICL checkpoint sub/v8 bundles (pinned commit, SHA-256 checked)
	tools/fetch_tabicl.sh sub/v8/weights
