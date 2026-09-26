"""Evaluation entry point for the anonymous review artifact."""

from flog.decoder import CoreImplementationUnavailable, flog_rollout


def main():
    try:
        flog_rollout()
    except CoreImplementationUnavailable as exc:
        print(exc)


if __name__ == "__main__":
    main()
