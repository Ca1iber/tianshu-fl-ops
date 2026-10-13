# SPDX-License-Identifier: Apache-2.0
"""Read-only reflection of one existing configuration table."""
from sqlalchemy.ext.automap import automap_base


def reflect_one(name, engine):
    base = automap_base()
    # The predicate preserves missing-table -> None behavior. A sequence
    # containing a missing table would make SQLAlchemy raise instead.
    base.prepare(autoload_with=engine,
                 reflection_options={'only': lambda table_name, metadata: table_name == name})
    return base.classes.get(name)
