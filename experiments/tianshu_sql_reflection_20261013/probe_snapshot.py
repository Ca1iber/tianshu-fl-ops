# SPDX-License-Identifier: Apache-2.0
"""Read-only local verification of a consistent Tianshu configuration snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    db=args.database.resolve();assert hashlib.sha256(db.read_bytes()).hexdigest()==args.sha256
    import sqlalchemy as sa
    from sqlalchemy.ext.automap import automap_base
    from reflect_one import reflect_one
    engine=sa.create_engine('sqlite://',creator=lambda:sqlite3.connect(f'file:{db}?mode=ro',uri=True))
    names=sa.inspect(engine).get_table_names()
    foreign_keys=[name for name in names if sa.inspect(engine).get_foreign_keys(name)]
    assert not foreign_keys,'Narrow reflection may recursively discover referenced tables'
    selected=[]
    for fragment in ['_index_jit_function_','mm_','radix','histogram','scan']:
        group=[name for name in names if fragment in name]
        if group:selected.append(group[0])
    selected=list(dict.fromkeys(selected+[names[0],names[len(names)//2],names[-1],'tianshu_missing_config_table']))
    rows=[]
    for name in selected:
        start=time.perf_counter();base=automap_base();base.prepare(engine);a=base.classes.get(name);old=time.perf_counter()-start
        start=time.perf_counter();b=reflect_one(name,engine);new=time.perf_counter()-start
        assert (a is None)==(b is None)
        count=0
        if a is not None:
            def columns(model):return [(c.name,str(c.type),c.primary_key,c.nullable) for c in model.__table__.columns]
            assert columns(a)==columns(b)
            assert [c.name for c in a.__table__.primary_key]==[c.name for c in b.__table__.primary_key]
            with engine.connect() as connection:
                left=connection.execute(sa.select(a.__table__).order_by(*a.__table__.primary_key).limit(32)).mappings().all()
                right=connection.execute(sa.select(b.__table__).order_by(*b.__table__.primary_key).limit(32)).mappings().all()
                assert [dict(x) for x in left]==[dict(x) for x in right]
                count=len(left)
        rows.append(dict(table=name,missing=a is None,columns_primary_key_rows_identical=True,
                         checked_rows=count,full_reflection_seconds=old,target_reflection_seconds=new))
    result=dict(complete=True,all_checks_passed=True,cpu_only=True,local_python=platform.python_version(),
                database_sha256=args.sha256,total_tables=len(names),foreign_key_tables=foreign_keys,
                source_database_modified=False,service_performance_not_measured=True,rows=rows)
    assert hashlib.sha256(db.read_bytes()).hexdigest()==args.sha256
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
