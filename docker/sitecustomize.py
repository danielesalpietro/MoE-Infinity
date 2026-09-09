# Applies MOE_LOG_LEVEL to the root logger for the serving image.
#
# api_server_v2 on upstream/main has no --log-level flag (it hardcodes
# uvicorn's log_level="info"), so there is no CLI path for turning up
# verbosity while diagnosing a slow or stuck model load. Python imports
# sitecustomize automatically at startup if it is importable, which gives us
# that knob without patching the server or the entrypoint's argv.
#
# Deliberately quiet: an unset or unparseable MOE_LOG_LEVEL leaves logging
# exactly as it was, so this file is a no-op in the default configuration.

import logging
import os

_level_name = (os.environ.get("MOE_LOG_LEVEL") or "").strip().upper()

if _level_name:
    _level = getattr(logging, _level_name, None)
    if isinstance(_level, int):
        logging.basicConfig(level=_level)
        logging.getLogger().setLevel(_level)
    else:
        # Never fail startup over a typo in an env var -- warn and carry on.
        logging.getLogger(__name__).warning(
            "MOE_LOG_LEVEL=%r is not a valid logging level; ignoring",
            _level_name,
        )
