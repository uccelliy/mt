#!/usr/bin/bash
# Safe Track-P submission front end.
#
# Active roster: R1/R2, R13-R16 and the two `gemma4_text` models. Two Gemma
# entries stay visible in `list` but blocked: the 26B mixture needs volta32
# (14.6 GB of 4-bit weights), and the 12B is `gemma4_unified_text`, whose
# sparse LM-head path failed its forward-path cross-check. Every formal
# output path contains both the readout protocol and the exact git commit,
# so old legal-option-argmax files cannot be resumed as greedy one-token
# files.
#
# Typical E3 sequence (use the official adapter for the infrastructure gate):
#
#   bash scripts/submit_roster.sh list
#   bash scripts/submit_roster.sh dry-run e3 smoke centaur8b
#   bash scripts/submit_roster.sh e3 smoke centaur8b
#   bash scripts/submit_roster.sh e3 l3 centaur8b
#   scancel <the printed L3 job id>
#   bash scripts/submit_roster.sh e3 l3 centaur8b   # resumes, merges, opens gate
#   bash scripts/submit_roster.sh e3 full centaur8b llama31_8b
#   bash scripts/submit_roster.sh e3 merge centaur8b llama31_8b
#
# A tag (or the explicit word `all`) is always required: there is no implicit
# submit-the-entire-roster default. Prefix a command with `dry-run` to print
# the exact sbatch command without submitting or changing gate state.

set -euo pipefail

usage() {
    cat >&2 <<EOF
usage:
  $0 list
  $0 dry-run {smoke|l3|full|merge} {tag ...|all}
  $0 {smoke|l3|full|merge} {tag ...|all}
  $0 dry-run e3 {smoke|l3|full|merge} {tag ...|all}
  $0 e3 {smoke|l3|full|merge} {tag ...|all}
EOF
}

# tag | HF model | adapter | constraint | status
ROSTER=(
    "llama31_8b|meta-llama/Llama-3.1-8B||volta16|active"
    "centaur8b|meta-llama/Llama-3.1-8B|marcelbinz/Llama-3.1-Centaur-8B-adapter|volta16|active"
    "gemma4_e2b|google/gemma-4-E2B-it||volta16|active"
    "gemma4_e4b|google/gemma-4-E4B-it||volta16|active"
    "gemma4_12b|google/gemma-4-12B-it||volta16|deferred"
    "gemma4_26b_a4b|google/gemma-4-26B-A4B-it||volta32|deferred"
    "llama32_1b|meta-llama/Llama-3.2-1B||volta16|active"
    "llama32_1b_instruct|meta-llama/Llama-3.2-1B-Instruct||volta16|active"
    "llama32_3b|meta-llama/Llama-3.2-3B||volta16|active"
    "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct||volta16|active"
)

list_roster() {
    printf '%-24s %-9s %s\n' "TAG" "STATUS" "MODEL"
    for entry in "${ROSTER[@]}"; do
        IFS='|' read -r tag model adapter constraint status <<< "$entry"
        printf '%-24s %-9s %s%s\n' "$tag" "$status" "$model" \
            "${adapter:+ + $adapter}"
    done
    echo
    echo "Deferred models are visible above but this launcher refuses to submit them."
}

DRY_RUN=0
COMMAND="${1:-}"
if [ "$COMMAND" = "dry-run" ]; then
    DRY_RUN=1
    shift
    COMMAND="${1:-}"
fi

if [ "$COMMAND" = "list" ]; then
    list_roster
    exit 0
fi
SUITE="full"
if [ "$COMMAND" = "e3" ]; then
    SUITE="e3"
    shift
    COMMAND="${1:-}"
fi
if [ "$COMMAND" != "smoke" ] && [ "$COMMAND" != "l3" ] && \
   [ "$COMMAND" != "full" ] && [ "$COMMAND" != "merge" ]; then
    usage
    exit 2
fi
shift || true
if [ "$#" -eq 0 ]; then
    echo "refusing implicit all-roster submission; name tags or use 'all'" >&2
    usage
    exit 2
fi

if [ "$DRY_RUN" -eq 0 ] && \
   [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "refusing submission from a dirty worktree: commit the exact protocol first" >&2
    exit 1
fi

if ! GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null); then
    echo "run this launcher from a git checkout" >&2
    exit 2
fi
GIT_SHORT="${GIT_COMMIT:0:12}"
CHOICE_READOUT="greedy-unconstrained-1token"
PROTOCOL_TAG="${MT_PROTOCOL_TAG:-trackp-greedy1tok-v1-${GIT_SHORT}}"
if [[ ! "$PROTOCOL_TAG" =~ ^trackp-greedy1tok-[A-Za-z0-9._-]+$ ]]; then
    echo "MT_PROTOCOL_TAG must start with trackp-greedy1tok- and be path-safe" >&2
    exit 2
fi
if [ -n "${MT_LOAD:-}" ] && [ "$MT_LOAD" != "4bit" ]; then
    echo "this formal Track-P launcher is pinned to MT_LOAD=4bit" >&2
    exit 2
fi
LOAD=4bit
OUTPUT_ROOT="${MT_OUTPUT_ROOT:-outputs/runs}"
PROTOCOL_ROOT="${OUTPUT_ROOT}/${PROTOCOL_TAG}/${LOAD}"
GATE_DIR="${PROTOCOL_ROOT}/.gate"
if [ "$SUITE" = "e3" ]; then
    GATE_FILE="${GATE_DIR}/e3-l3-passed"
    EXPECTED_GATE="${PROTOCOL_TAG}|${GIT_COMMIT}|${CHOICE_READOUT}|e3"
else
    GATE_FILE="${GATE_DIR}/l3-passed"
    EXPECTED_GATE="${PROTOCOL_TAG}|${GIT_COMMIT}|${CHOICE_READOUT}"
fi
SBATCH_BIN="${MT_SBATCH:-sbatch}"
# Both gates are dominated by model loading, not by scoring. Measured on
# iris: 75 smoke sessions score in 3.5 minutes, while loading the 4-bit 8B
# base plus adapter took 36 seconds on a node whose page cache was already
# warm, 14 minutes on a lukewarm one, and over 50 minutes on a cold one --
# which is what made the one-hour smoke time out on iris-185 after doing no
# scoring at all. Three hours covers a cold load with room to spare; the
# jobs still exit as soon as they finish, so the only cost is queue
# priority. `full` is unaffected: a cold load is 2% of its 47-hour budget.
GATE_TIME="${MT_GATE_TIME:-03:00:00}"

requested=("$@")
if [ "${#requested[@]}" -eq 1 ] && [ "${requested[0]}" = "all" ]; then
    requested=()
    for entry in "${ROSTER[@]}"; do
        IFS='|' read -r tag model adapter constraint status <<< "$entry"
        [ "$status" = "active" ] && requested+=("$tag")
    done
elif printf '%s\n' "${requested[@]}" | grep -qx all; then
    echo "'all' cannot be combined with named tags" >&2
    exit 2
fi
if [ "$COMMAND" = "l3" ] && [ "${#requested[@]}" -ne 1 ]; then
    echo "L3 is a single-model infrastructure gate; name exactly one tag" >&2
    exit 2
fi

seen_tags="|"
for requested_tag in "${requested[@]}"; do
    case "$seen_tags" in
        *"|${requested_tag}|"*)
            echo "duplicate roster tag: $requested_tag" >&2
            exit 2
            ;;
    esac
    seen_tags="${seen_tags}${requested_tag}|"
done

lookup_entry() {
    local wanted="$1"
    local entry tag model adapter constraint status
    for entry in "${ROSTER[@]}"; do
        IFS='|' read -r tag model adapter constraint status <<< "$entry"
        if [ "$tag" = "$wanted" ]; then
            printf '%s\n' "$entry"
            return 0
        fi
    done
    return 1
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

submit() {
    local output
    echo "submit:" >&2
    print_command "$@" >&2
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN"
        return 0
    fi
    output=$("$@")
    printf '%s\n' "$output"
}

assert_full_gate() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  dry-run note: formal full submission requires $GATE_FILE" >&2
        return 0
    fi
    if [ ! -f "$GATE_FILE" ]; then
        echo "formal full is blocked: L3 gate is missing: $GATE_FILE" >&2
        echo "run/cancel/resubmit L3 as shown at the top of this script" >&2
        exit 1
    fi
    local actual
    actual=$(tr -d '\r\n' < "$GATE_FILE")
    if [ "$actual" != "$EXPECTED_GATE" ]; then
        echo "formal full is blocked: L3 gate belongs to another protocol/commit" >&2
        echo "  expected: $EXPECTED_GATE" >&2
        echo "  actual:   $actual" >&2
        exit 1
    fi
}

assert_smoke_gate() {
    local run_root="$1"
    local tag="$2"
    local smoke_gate="${run_root}/.smoke-passed"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  dry-run note: $tag full/L3 requires $smoke_gate" >&2
        return 0
    fi
    if [ ! -f "$smoke_gate" ]; then
        echo "$tag is blocked: protocol-matched smoke gate is missing: $smoke_gate" >&2
        if [ "$SUITE" = "e3" ]; then
            echo "run '$0 e3 smoke $tag' first" >&2
        else
            echo "run '$0 smoke $tag' first" >&2
        fi
        exit 1
    fi
    local actual
    actual=$(tr -d '\r\n' < "$smoke_gate")
    if [ "$actual" != "$EXPECTED_GATE" ]; then
        echo "$tag is blocked: smoke gate belongs to another protocol/commit" >&2
        exit 1
    fi
}

job_state_file() {
    local tag="$1"
    local role="$2"
    local prefix=""
    [ "$SUITE" = "e3" ] && prefix="e3-"
    case "$role" in
        l3-score) printf '%s/%sl3-%s.jobs\n' "$GATE_DIR" "$prefix" "$tag" ;;
        l3-merge) printf '%s/%sl3-accept-%s.jobs\n' "$GATE_DIR" "$prefix" "$tag" ;;
        smoke) printf '%s/%ssmoke-%s.jobs\n' "$GATE_DIR" "$prefix" "$tag" ;;
        full-score) printf '%s/%sfull-%s.jobs\n' "$GATE_DIR" "$prefix" "$tag" ;;
        full-merge) printf '%s/%smerge-%s.jobs\n' "$GATE_DIR" "$prefix" "$tag" ;;
        *) echo "unknown job-state role: $role" >&2; return 2 ;;
    esac
}

query_live_slurm_job() {
    local job_id="$1"
    local queue
    # Query the live queue as a whole. Some Slurm versions return a non-zero
    # status for `squeue -j <completed-id>`, which is not a controller failure
    # and must not make an old state-file entry block every later submission.
    if ! queue=$(squeue -h -o '%A|%T'); then
        return 1
    fi
    awk -F '|' -v wanted="$job_id" '
        {
            id = $1
            state = $2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", id)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", state)
            if (id == wanted) {
                print state
                exit
            }
        }
    ' <<< "$queue"
}

assert_job_inactive() {
    local state_file="$1"
    local label="$2"
    [ "$DRY_RUN" -eq 1 ] && return 0
    [ -f "$state_file" ] || return 0
    local previous
    previous=$(tail -n 1 "$state_file")
    [ -n "$previous" ] || return 0
    local state
    if ! state=$(query_live_slurm_job "$previous"); then
        echo "cannot query prior $label job $previous" >&2
        exit 1
    fi
    if [ -n "$state" ]; then
        echo "$label job $previous is still active ($state); no job was submitted" >&2
        exit 1
    fi
}

record_job() {
    local state_file="$1"
    local job_id="$2"
    mkdir -p "$GATE_DIR"
    printf '%s\n' "$job_id" >> "$state_file"
}

l3_attempt() {
    local tag="$1"
    local state_file
    state_file=$(job_state_file "$tag" l3-score)
    local state_prefix
    state_prefix=$(basename "$state_file" .jobs)
    local cancel_marker="${GATE_DIR}/${state_prefix}.cancel-verified"
    local previous=""
    if [ -f "$state_file" ]; then
        previous=$(tail -n 1 "$state_file")
    fi
    if [ -z "$previous" ]; then
        printf '1|%s|%s\n' "$state_file" "$cancel_marker"
        return 0
    fi
    if [ "$DRY_RUN" -eq 0 ]; then
        local live_state
        if ! live_state=$(query_live_slurm_job "$previous"); then
            echo "cannot query prior L3 job $previous" >&2
            exit 1
        fi
        if [ -n "$live_state" ]; then
            echo "L3 job $previous is still active ($live_state); " \
                 "scancel it before resubmitting" >&2
            exit 1
        fi
    fi
    local verified_cancel=""
    if [ -f "$cancel_marker" ]; then
        verified_cancel=$(tr -d '\r\n' < "$cancel_marker")
    fi
    if [ "$verified_cancel" = "$previous" ]; then
        local count
        count=$(wc -l < "$state_file")
        printf '%s|%s|%s\n' "$((count + 1))" "$state_file" "$cancel_marker"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '2|%s|%s\n' "$state_file" "$cancel_marker"
        return 0
    fi
    local state
    state=$(sacct -n -X -j "$previous" --format=State 2>/dev/null \
        | awk 'NF {print $1; exit}')
    case "$state" in
        CANCELLED*)
            mkdir -p "$GATE_DIR"
            printf '%s\n' "$previous" > "$cancel_marker"
            ;;
        *)
            echo "L3 gate requires an observed scancel; job $previous state is '${state:-unknown}'" >&2
            exit 1
            ;;
    esac
    local count
    count=$(wc -l < "$state_file")
    printf '%s|%s|%s\n' "$((count + 1))" "$state_file" "$cancel_marker"
}

count_success_commits() {
    local commit_dir="$1"
    local count=0
    local marker
    [ -d "$commit_dir" ] || { printf '0\n'; return 0; }
    for marker in "$commit_dir"/*.json; do
        [ -f "$marker" ] || continue
        if ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"success"' "$marker"; then
            echo "L3 resume proof found a non-success marker: $marker" >&2
            return 1
        fi
        count=$((count + 1))
    done
    printf '%s\n' "$count"
}

prepare_l3_resume_proof() {
    local run_dir="$1"
    local cancel_marker="$2"
    local proof_file="${cancel_marker%.cancel-verified}.resume-proof.tsv"
    if [ -f "$proof_file" ]; then
        if [ "$(wc -l < "$proof_file")" -ne 5 ]; then
            echo "malformed existing L3 resume proof: $proof_file" >&2
            exit 1
        fi
        printf '%s\n' "$proof_file"
        return 0
    fi
    if [ "$SUITE" != "e3" ]; then
        echo "atomic L3 resume proof is currently required through the E3 runner" >&2
        echo "submit this experiment with '$0 e3 l3 <tag>'" >&2
        exit 1
    fi
    local temporary="${proof_file}.tmp.$$"
    mkdir -p "$GATE_DIR"
    printf 'shard\tcommitted_sessions\tprediction_rows\n' > "$temporary"
    local shard commit_count prediction prediction_rows
    for shard in 0 1 2 3; do
        if ! commit_count=$(count_success_commits \
            "${run_dir}/session_commits_shard${shard}"); then
            unlink "$temporary" 2>/dev/null || true
            return 1
        fi
        prediction="${run_dir}/predictions_shard${shard}.csv"
        if [ ! -f "$prediction" ]; then
            echo "L3 was cancelled before shard $shard produced predictions: $prediction" >&2
            unlink "$temporary" 2>/dev/null || true
            exit 1
        fi
        prediction_rows=$(($(wc -l < "$prediction") - 1))
        if [ "$commit_count" -le 0 ] || [ "$prediction_rows" -le 0 ]; then
            echo "L3 was cancelled before shard $shard committed a session" >&2
            unlink "$temporary" 2>/dev/null || true
            exit 1
        fi
        if [ "$commit_count" -ge 50 ]; then
            echo "shard $shard already completed all 50 L3 sessions; " \
                 "the required four-shard resume was not exercised" >&2
            unlink "$temporary" 2>/dev/null || true
            exit 1
        fi
        printf '%s\t%s\t%s\n' "$shard" "$commit_count" "$prediction_rows" \
            >> "$temporary"
    done
    mv "$temporary" "$proof_file"
    printf '%s\n' "$proof_file"
}

# Validate the complete request before the first sbatch. In particular, an
# `all` command must never submit a prefix of the roster and then discover a
# missing gate on a later model.
for requested_tag in "${requested[@]}"; do
    if ! entry=$(lookup_entry "$requested_tag"); then
        echo "unknown roster tag: $requested_tag (run '$0 list')" >&2
        exit 2
    fi
    IFS='|' read -r tag model adapter constraint status <<< "$entry"
    if [ "$status" != "active" ]; then
        echo "$tag is deferred; no job was submitted" >&2
        exit 1
    fi
    suite_run_root="${PROTOCOL_ROOT}/${tag}"
    [ "$SUITE" = "e3" ] && suite_run_root="${suite_run_root}/e3"
    case "$COMMAND" in
        smoke)
            assert_job_inactive "$(job_state_file "$tag" smoke)" "$tag smoke"
            ;;
        l3)
            assert_smoke_gate "$suite_run_root" "$tag"
            assert_job_inactive "$(job_state_file "$tag" l3-score)" "$tag L3 score"
            assert_job_inactive "$(job_state_file "$tag" l3-merge)" "$tag L3 acceptance"
            ;;
        full)
            assert_smoke_gate "$suite_run_root" "$tag"
            assert_full_gate
            assert_job_inactive "$(job_state_file "$tag" full-score)" "$tag full score"
            assert_job_inactive "$(job_state_file "$tag" full-merge)" "$tag full merge"
            ;;
        merge)
            assert_job_inactive "$(job_state_file "$tag" full-score)" "$tag full score"
            assert_job_inactive "$(job_state_file "$tag" full-merge)" "$tag full merge"
            if [ "$DRY_RUN" -eq 0 ] && [ ! -d "${suite_run_root}/full" ]; then
                echo "$tag has no full run directory to merge: ${suite_run_root}/full" >&2
                exit 1
            fi
            ;;
    esac
done

for requested_tag in "${requested[@]}"; do
    if ! entry=$(lookup_entry "$requested_tag"); then
        echo "unknown roster tag: $requested_tag (run '$0 list')" >&2
        exit 2
    fi
    IFS='|' read -r tag model adapter constraint status <<< "$entry"
    if [ "$status" != "active" ]; then
        echo "$tag is deferred; no job was submitted" >&2
        exit 1
    fi

    run_root="${PROTOCOL_ROOT}/${tag}"
    suite_run_root="$run_root"
    job_prefix="mt"
    if [ "$SUITE" = "e3" ]; then
        suite_run_root="${run_root}/e3"
        job_prefix="mt-e3"
    fi
    # Deliberately omit ALL: formal jobs receive only the variables below,
    # so an old MT_SMOKE/MT_LIMIT/MT_MAX_CHARS in the login shell cannot
    # silently change a full run. Slurm still supplies its SLURM_* variables;
    # site configuration comes from the versioned hpc_env.sh.
    notify_email="${MT_NOTIFY_EMAIL:-${SBATCH_MAIL_USER:-}}"
    # Pass the address explicitly. Exporting SBATCH_MAIL_USER did populate the
    # in-process progress notifier but produced no BEGIN/END/FAIL mail, and the
    # form verified on this cluster (docs/Server Alarm.md) is `sbatch
    # --mail-user=<you>@uni.lu`. Nothing here is left to shell environment.
    mail_args=()
    if [ -n "$notify_email" ]; then
        mail_args=(--mail-user="$notify_email")
    fi
    common_export="USER=${USER:?},HOME=${HOME:?},MT_NOTIFY_EMAIL=${notify_email}"
    common_export+=",MT_MODEL=${model},MT_ADAPTER=${adapter},MT_TAG=${tag}"
    common_export+=",MT_LOAD=${LOAD},MT_PROTOCOL_TAG=${PROTOCOL_TAG}"
    common_export+=",MT_PROTOCOL_COMMIT=${GIT_COMMIT},MT_RUN_ROOT=${suite_run_root}"
    common_export+=",MT_CHOICE_READOUT=${CHOICE_READOUT}"
    common_export+=",MT_SUITE=${SUITE}"

    case "$COMMAND" in
        smoke)
            export_vars="${common_export},MT_RUN_MODE=smoke,MT_NGPU=1"
            result=$(submit "$SBATCH_BIN" --parsable ${mail_args[@]+"${mail_args[@]}"} --constraint="$constraint" \
                --nodes=1 --ntasks-per-node=1 --cpus-per-task=7 \
                --gpus-per-task=1 --mem=64G --time="$GATE_TIME" \
                --job-name="${job_prefix}-smoke-${tag}" --export="$export_vars" \
                scripts/score_model.slurm)
            if [ "$DRY_RUN" -eq 1 ]; then
                printf '%s\n' "$result"
            else
                score_job="${result%%;*}"
                record_job "$(job_state_file "$tag" smoke)" "$score_job"
                echo "smoke job: $score_job"
            fi
            ;;
        l3)
            IFS='|' read -r attempt state_file cancel_marker \
                <<< "$(l3_attempt "$tag")"
            l3_proof=""
            proof_ready=1
            if [ "$attempt" -ge 2 ] && [ "$DRY_RUN" -eq 0 ]; then
                if [ "$SUITE" != "e3" ]; then
                    echo "the atomic interrupted-resume gate is implemented by the E3 runner" >&2
                    echo "use '$0 e3 l3 $tag' for the unified full + E3 experiment" >&2
                    exit 1
                fi
                if ! l3_proof=$(prepare_l3_resume_proof "${suite_run_root}/l3" \
                    "$cancel_marker"); then
                    proof_ready=0
                    l3_proof=""
                    echo "submitting another L3 collection attempt; cancel it only after " \
                         "all four shards have committed at least one session" >&2
                fi
            fi
            export_vars="${common_export},MT_RUN_MODE=l3,MT_NGPU=4"
            export_vars+=",MT_L3_LIMIT=200,MT_L3_ATTEMPT=${attempt}"
            result=$(submit "$SBATCH_BIN" --parsable ${mail_args[@]+"${mail_args[@]}"} --constraint="$constraint" \
                --nodes=1 --ntasks-per-node=1 --cpus-per-task=28 \
                --gpus-per-task=4 --mem=128G --time="$GATE_TIME" \
                --job-name="${job_prefix}-l3-${tag}" --export="$export_vars" \
                scripts/score_model.slurm)
            if [ "$DRY_RUN" -eq 1 ]; then
                if [ "$attempt" -eq 1 ]; then
                    echo "  first L3 submission: scancel it, then submit the same command again"
                else
                    echo "  resumed L3 would get an afterok merge/acceptance job"
                fi
                continue
            fi
            score_job="${result%%;*}"
            record_job "$state_file" "$score_job"
            echo "L3 score job: $score_job"
            if [ "$attempt" -eq 1 ] || [ "$proof_ready" -eq 0 ]; then
                echo "scancel $score_job after all four shards have grown; then rerun this L3 command"
                continue
            fi
            merge_export="USER=${USER:?},HOME=${HOME:?}"
            merge_export+=",MT_RUN_DIRS=${suite_run_root}/l3,MT_NGPU=4"
            merge_export+=",MT_EXPECTED_SESSIONS=200,MT_GATE_FILE=${GATE_FILE}"
            merge_export+=",MT_GATE_VALUE=${EXPECTED_GATE}"
            merge_export+=",MT_L3_PROOF_FILE=${l3_proof}"
            merge_result=$(submit "$SBATCH_BIN" --parsable ${mail_args[@]+"${mail_args[@]}"} \
                --dependency="afterok:${score_job}" \
                --job-name="${job_prefix}-l3-accept-${tag}" --export="$merge_export" \
                scripts/merge_shards.slurm)
            merge_job="${merge_result%%;*}"
            record_job "$(job_state_file "$tag" l3-merge)" "$merge_job"
            echo "L3 acceptance job: $merge_job (opens full gate only on success)"
            ;;
        full)
            export_vars="${common_export},MT_RUN_MODE=full,MT_NGPU=4"
            result=$(submit "$SBATCH_BIN" --parsable ${mail_args[@]+"${mail_args[@]}"} --constraint="$constraint" \
                --nodes=1 --ntasks-per-node=1 --cpus-per-task=28 \
                --gpus-per-task=4 --mem=128G --time=1-23:00:00 \
                --job-name="${job_prefix}-score-${tag}" --export="$export_vars" \
                scripts/score_model.slurm)
            if [ "$DRY_RUN" -eq 1 ]; then
                printf '%s\n' "$result"
            else
                score_job="${result%%;*}"
                record_job "$(job_state_file "$tag" full-score)" "$score_job"
                echo "full score job: $score_job"
            fi
            ;;
        merge)
            merge_export="USER=${USER:?},HOME=${HOME:?}"
            merge_export+=",MT_RUN_DIRS=${suite_run_root}/full,MT_NGPU=4"
                        # Each run's launcher.json already records the count it targeted
            # after the deferred experiments were removed; re-deriving it here
            # would need a Python the login node does not have.
            merge_export+=",MT_EXPECTED_SESSIONS=auto"
            result=$(submit "$SBATCH_BIN" --parsable ${mail_args[@]+"${mail_args[@]}"} --job-name="${job_prefix}-merge-${tag}" \
                --export="$merge_export" scripts/merge_shards.slurm)
            if [ "$DRY_RUN" -eq 1 ]; then
                printf '%s\n' "$result"
            else
                merge_job="${result%%;*}"
                record_job "$(job_state_file "$tag" full-merge)" "$merge_job"
                echo "merge job: $merge_job"
            fi
            ;;
    esac
done
