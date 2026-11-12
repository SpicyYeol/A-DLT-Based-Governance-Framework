# -*- coding: utf-8 -*-
"""Report writers.

Produces the Structure Trust Factors deliverable: the taxonomy with its
``Detailed Category`` and ``Control points`` columns filled in, plus the live
status and score for each of the 36 sub-categories.

The writers are split across the modules of this package by the
kind of deliverable they produce, and everything is re-exported
here, so `report.write_baseline_csv()` and every other call site
is unchanged.
"""
from __future__ import annotations

from . import verdicts as verdicts  # noqa: F401
from . import headline_tables as headline_tables  # noqa: F401
from . import experiment_tables as experiment_tables  # noqa: F401
from . import summaries as summaries  # noqa: F401

from .verdicts import *  # noqa: F401,F403
from .headline_tables import *  # noqa: F401,F403
from .experiment_tables import *  # noqa: F401,F403
from .summaries import *  # noqa: F401,F403
