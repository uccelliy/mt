#!/usr/bin/bash
# Probe the two mail paths needed by the server reminder design:
#
#   1. direct mail from the node running this script; and
#   2. Slurm's native BEGIN/END/FAIL notifications, plus direct mail from the
#      compute node allocated to a tiny probe job.
#
# No credentials are read or stored. A zero exit status from a mail command
# only means that the local mail system accepted the message; delivery must be
# confirmed in the recipient inbox.

set -uo pipefail

export LC_ALL=C
umask 077

PROGRAM_NAME="${0##*/}"
COMMAND_TIMEOUT_SECONDS=20

usage() {
    cat <<EOF
Usage:
  $PROGRAM_NAME --check-only
  $PROGRAM_NAME --send-test EMAIL
  $PROGRAM_NAME --submit-slurm-test EMAIL
  $PROGRAM_NAME --all EMAIL

Modes:
  --check-only         Print available commands without sending mail or submitting a job.
                       This is the default.
  --send-test          Send one direct test message from the current node.
  --submit-slurm-test  Submit a tiny job that tests Slurm notifications and direct mail
                       from a compute node.
  --all                Run the capability check, direct test, and Slurm test.

Optional environment overrides for --submit-slurm-test:
  MT_MAIL_PROBE_PARTITION   Slurm partition (default: interactive)
  MT_MAIL_PROBE_QOS         Slurm QOS (default: debug)

Examples:
  bash scripts/hpc_mail_probe.sh --check-only
  bash scripts/hpc_mail_probe.sh --send-test name.surname@uni.lu
  bash scripts/hpc_mail_probe.sh --submit-slurm-test name.surname@uni.lu
EOF
}

error() {
    printf 'ERROR: %s\n' "$*" >&2
}

validate_recipient() {
    local recipient="$1"

    # A leading dash would be parsed as an option by mailx and mail, which
    # take the recipient as a positional argument. The character class below
    # allows '-' inside the local part, so this has to be checked separately.
    if [ "${recipient#-}" != "$recipient" ]; then
        error "invalid email address: must not start with '-'"
        return 1
    fi
    if [[ ! "$recipient" =~ ^[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}$ ]]; then
        error "invalid email address"
        return 1
    fi
}

run_limited() {
    local timeout_path=""

    if timeout_path=$(type -P timeout); then
        "$timeout_path" "$COMMAND_TIMEOUT_SECONDS" "$@"
    else
        error "timeout is unavailable; refusing to run an unbounded external command"
        return 125
    fi
}

find_sendmail() {
    local sendmail_path=""

    if sendmail_path=$(type -P sendmail); then
        printf '%s\n' "$sendmail_path"
    elif [ -x /usr/sbin/sendmail ]; then
        printf '%s\n' /usr/sbin/sendmail
    elif [ -x /usr/lib/sendmail ]; then
        printf '%s\n' /usr/lib/sendmail
    else
        return 1
    fi
}

command_path_or_absent() {
    local name="$1"
    local path=""

    if path=$(type -P "$name"); then
        printf '%s\n' "$path"
    else
        printf '%s\n' absent
    fi
}

check_capabilities() {
    local context="outside a Slurm allocation"
    local sendmail_path="absent"

    if [ -n "${SLURM_JOB_ID:-}" ]; then
        context="Slurm compute job"
    fi
    if sendmail_path=$(find_sendmail); then
        :
    else
        sendmail_path="absent"
    fi

    printf 'Mail probe capability check (no message will be sent)\n'
    printf 'Context:  %s\n' "$context"
    printf 'bash:     %s\n' "${BASH_VERSION:-unknown}"
    printf 'timeout:  %s\n' "$(command_path_or_absent timeout)"
    printf 'sendmail: %s\n' "$sendmail_path"
    printf 'mailx:    %s\n' "$(command_path_or_absent mailx)"
    printf 'mail:     %s\n' "$(command_path_or_absent mail)"
    printf 'sbatch:   %s\n' "$(command_path_or_absent sbatch)"
}

send_direct_probe() {
    local recipient="$1"
    local context="outside a Slurm allocation"
    local job_id="not running under Slurm"
    local mailer=""
    local probe_id=""
    local subject=""
    local body=""

    if [ -n "${SLURM_JOB_ID:-}" ]; then
        context="Slurm compute job"
        job_id="$SLURM_JOB_ID"
    fi

    probe_id="${SLURM_JOB_ID:-manual}-$(date -u '+%Y%m%dT%H%M%SZ')-$$"
    subject="[mt] direct mail probe $probe_id"
    body="This is an mt server mail probe.

Probe ID: $probe_id
Delivery path: local mail command
Execution context: $context
Slurm Job ID: $job_id
UTC time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')

Receiving this message confirms that direct mail was accepted from this context."

    printf 'Testing direct mail from: %s\n' "$context"
    if mailer=$(find_sendmail); then
        printf 'Backend: %s -t -oi\n' "$mailer"
        if {
            printf 'To: %s\n' "$recipient"
            printf 'Subject: %s\n' "$subject"
            printf 'MIME-Version: 1.0\n'
            printf 'Content-Type: text/plain; charset=UTF-8\n'
            printf '\n%s\n' "$body"
        } | run_limited "$mailer" -t -oi; then
            printf 'Direct message accepted by %s; confirm delivery in the inbox.\n' "$mailer"
            return 0
        fi
        error "$mailer rejected or timed out while accepting the message"
        return 1
    fi

    for candidate in mailx mail; do
        if mailer=$(type -P "$candidate"); then
            printf 'Backend: %s -s <subject> <recipient>\n' "$mailer"
            if printf '%s\n' "$body" | run_limited "$mailer" -s "$subject" "$recipient"; then
                printf 'Direct message accepted by %s; confirm delivery in the inbox.\n' "$mailer"
                return 0
            fi
            error "$mailer rejected or timed out while accepting the message"
            return 1
        fi
    done

    error "none of sendmail, mailx, or mail is available in this context"
    return 1
}

submit_slurm_probe() {
    local recipient="$1"
    local script_dir=""
    local script_path=""
    local partition="${MT_MAIL_PROBE_PARTITION:-interactive}"
    local qos="${MT_MAIL_PROBE_QOS:-debug}"
    local sbatch_path=""
    local submission=""
    local job_id=""

    if [ -n "${SLURM_JOB_ID:-}" ]; then
        error "--submit-slurm-test must be run outside an existing Slurm allocation"
        return 1
    fi
    if ! sbatch_path=$(type -P sbatch); then
        error "sbatch is unavailable; run --submit-slurm-test from an ULHPC login node"
        return 1
    fi

    script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || return 1
    script_path="$script_dir/$PROGRAM_NAME"
    export MT_MAIL_PROBE_EMAIL="$recipient"
    export MT_MAIL_PROBE_SCRIPT="$script_path"

    printf 'Submitting tiny Slurm mail probe: partition=%s qos=%s\n' "$partition" "$qos"
    if ! submission=$(run_limited "$sbatch_path" --parsable \
        --job-name=mt-mail-probe \
        --output=mt-mail-probe-%j.out \
        --error=mt-mail-probe-%j.err \
        --partition="$partition" \
        --qos="$qos" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=1 \
        --mem=128M \
        --time=00:02:00 \
        --mail-user="$recipient" \
        --mail-type=BEGIN,END,FAIL \
        --export=ALL,MT_MAIL_PROBE_EMAIL,MT_MAIL_PROBE_SCRIPT \
        --wrap='bash "$MT_MAIL_PROBE_SCRIPT" --send-test "$MT_MAIL_PROBE_EMAIL"' 2>&1); then
        error "Slurm probe submission failed: $submission"
        return 1
    fi

    job_id="${submission%%;*}"
    printf 'Submitted Job ID: %s\n' "$job_id"
    printf 'Expected messages:\n'
    printf '  1. Slurm BEGIN notification.\n'
    printf '  2. Direct message sent from inside the compute job.\n'
    printf '  3. Slurm END notification.\n'
    printf 'If compute-node direct mail fails, the job exits non-zero and Slurm should send FAIL.\n'
    printf 'Inspect with:\n'
    printf '  squeue -j %s\n' "$job_id"
    printf '  sacct -X -j %s --format=JobID,State,ExitCode,Elapsed\n' "$job_id"
}

main() {
    local mode="check"
    local recipient=""
    local status=0

    case "${1:-}" in
        "")
            mode="check"
            ;;
        --check-only)
            mode="check"
            shift
            ;;
        --send-test)
            mode="direct"
            shift
            ;;
        --submit-slurm-test)
            mode="slurm"
            shift
            ;;
        --all)
            mode="all"
            shift
            ;;
        -h|--help)
            usage
            return 0
            ;;
    esac

    if [ "$mode" = "check" ]; then
        if [ "$#" -ne 0 ]; then
            usage >&2
            return 2
        fi
        check_capabilities
        return 0
    fi

    if [ "$#" -ne 1 ]; then
        usage >&2
        return 2
    fi
    recipient="$1"
    validate_recipient "$recipient" || return 2

    case "$mode" in
        direct)
            send_direct_probe "$recipient"
            ;;
        slurm)
            submit_slurm_probe "$recipient"
            ;;
        all)
            check_capabilities
            printf '\n'
            send_direct_probe "$recipient" || status=1
            printf '\n'
            submit_slurm_probe "$recipient" || status=1
            return "$status"
            ;;
    esac
}

main "$@"
