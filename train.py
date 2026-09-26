"""Training entry point for the anonymous review artifact.

The repository intentionally withholds the protected fairness-guided decoder and
recovery implementation. The full executable training pipeline will accompany
the complete public release.
"""

from flog.decoder import CoreImplementationUnavailable, flog_rollout


def main():
    try:
        flog_rollout()
    except CoreImplementationUnavailable as exc:
        print(exc)


if __name__ == "__main__":
    main()
