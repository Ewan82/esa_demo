MBA_TSTART := 20170501
MBA_TEND   := 20170731

SCHEDULE_S1A := input/target_schedule_S1A_2017.asc
SCHEDULE_S1B := input/target_schedule_S1B_2017.asc
SCHEDULE_S2A := input/target_schedule_S2A_2017.asc
SCHEDULE_S2B := input/target_schedule_S2B_2017.asc
MBA_SCHEDULE_FILE := target_schedule_mba.txt

MBAARGXTRA = --no_targets

$(MBA_SCHEDULE_FILE):
	@if [ "x$(SCHEDULE_LST)" = "x" ]; \
         then \
           cat $(INPUT_DIR)/target_schedule_2017.txt > $@; \
         else \
           cat $(INPUT_DIR)/target_schedule_2017.txt $(SCHEDULE_LST) > $@; \
         fi;
.PHONY: $(MBA_SCHEDULE_FILE)

mba_pre: $(MBA_SCHEDULE_FILE)
	( args=""; \
          args="$${args} --time_start $(MBA_TSTART) --time_end $(MBA_TEND)"; \
          args="$${args} --target_schedule $(MBA_SCHEDULE_FILE)"; \
          args="$${args} --use_generic_prior"; \
	  if [ "x$(MISSION_LST)" != "x" ]; \
          then \
	    args="$${args} --mission_lst $(MISSION_LST)"; \
          fi; \
          args="$${args} $(PREARGXTRA)"; \
	  $(RS_PRE_TOOL) pre_synthetic $${args} 2>&1 | tee mk.$@.log; \
        )
.PHONY: mba_pre
EXPCLEAN_FILES += mk.mba_pre.log

mba_synthetic: simulate.x $(RETRCONFIG_FILE) $(RETRPRIOR_FILE) $(SITE_FILE)
	./$< $(MBA_S2_FAILURE) 2>&1 | tee mk.$@.log
	if [ -f sensimul_s1.nc ]; then mv sensimul_s1.nc obs_s1.nc; fi
	if [ -f sensimul_s2.nc ]; then mv sensimul_s2.nc obs_s2.nc; fi
EXPCLEAN_FILES += mk.mba_synthetic.log


#-- (1) No Observation
MBA_noobs: simulate.x runua.x
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST= SCHEDULE_LST="$(SCHEDULE_S1A) $(SCHEDULE_S1B) $(SCHEDULE_S2A) $(SCHEDULE_S2B)"
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_noobs
MBA_EXPLIST += MBA_noobs

#-- (2) Only optical data (S2)
MBA_only-s2: simulate.x  runua.x
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S2A S2B" SCHEDULE_LST="$(SCHEDULE_S1A) $(SCHEDULE_S1B)"
	$(MAKE) mba_synthetic
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_only-s2
MBA_EXPLIST += MBA_only-s2

#-- (3) Only SAR data (S1)
MBA_only-s1: simulate.x  runua.x
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S1A S1B" SCHEDULE_LST="$(SCHEDULE_S2A) $(SCHEDULE_S2B)"
	$(MAKE) mba_synthetic
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_only-s1
MBA_EXPLIST += MBA_only-s1

#-- (4) Both data (S1+S2)
MBA_s1s2_schedule:
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S1A S1B S2A S2B"

MBA_s1s2: simulate.x  runua.x
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S1A S1B S2A S2B"
	$(MAKE) mba_synthetic
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_s1s2
MBA_EXPLIST += MBA_s1s2

#-- no S1A
MBA_no-S1A: simulate.x  runua.x
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S1B S2A S2B" SCHEDULE_LST="$(SCHEDULE_S1A)"
	$(MAKE) mba_synthetic
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_no-S1A
MBA_EXPLIST += MBA_no-S1A

#-- no S1B
MBA_no-S1B: simulate.x  runua.x
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S1A S2A S2B" SCHEDULE_LST="$(SCHEDULE_S1B)"
	$(MAKE) mba_synthetic
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_no-S1B
MBA_EXPLIST += MBA_no-S1B

#-- no S2A
MBA_no-S2A: simulate.x  runua.x
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S1A S1B S2B" SCHEDULE_LST="$(SCHEDULE_S2A)"
	$(MAKE) mba_synthetic
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_no-S2A
MBA_EXPLIST += MBA_no-S2A

#-- no S2B
MBA_no-S2B: simulate.x  runua.x
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S1A S1B S2A" SCHEDULE_LST="$(SCHEDULE_S2B)"
	$(MAKE) mba_synthetic
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_no-S2B
MBA_EXPLIST += MBA_no-S2B

MBA_S2A-no-swnir:
	$(MAKE) clean_exp
	$(MAKE) mba_pre
	$(MAKE) mba_synthetic MBA_S2_FAILURE="--s2a_no_swnir"
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_S2A-swnir
MBA_EXPLIST += MBA_S2A-no-swnir

MBA_S2A-no-visnir:
	$(MAKE) clean_exp
	$(MAKE) mba_pre
	$(MAKE) mba_synthetic MBA_S2_FAILURE="--s2a_no_visnir"
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_S2A-visnir
MBA_EXPLIST += MBA_S2A-no-visnir

MBA_S2B-no-S2A-no-swnir:
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S1A S1B S2A" SCHEDULE_LST="$(SCHEDULE_S2B)"
	$(MAKE) mba_synthetic MBA_S2_FAILURE="--s2a_no_swnir"
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_S2B-no-S2A-swnir
MBA_EXPLIST += MBA_S2B-no-S2A-no-swnir

MBA_S2B-no-S2A-no-visnir:
	$(MAKE) clean_exp
	$(MAKE) mba_pre MISSION_LST="S1A S1B S2A" SCHEDULE_LST="$(SCHEDULE_S2B)"
	$(MAKE) mba_synthetic MBA_S2_FAILURE="--s2a_no_visnir"
	$(MAKE) mba
	$(MAKE) save_exp SAVDIR=resdir_$@
	mv $(MBA_SCHEDULE_FILE) resdir_$@
.PHONY: MBA_S2B-no-S2A-visnir
MBA_EXPLIST += MBA_S2B-no-S2A-no-visnir

mba_all:
	@for exp in `echo $(MBA_EXPLIST)`; \
         do \
           $(MAKE) $${exp}; \
         done; \
         $(MAKE) clean_exp
