# SPDX-License-Identifier: Apache-2.0
"""CPU-only config reflection equivalence; no FlagGems/torch import."""
import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    import sqlalchemy as sa
    from sqlalchemy.ext.automap import automap_base
    from sqlalchemy.orm import Session
    from reflect_one import reflect_one
    engine=sa.create_engine('sqlite:///'+str(root/'synthetic_configs.db'))
    metadata=sa.MetaData()
    for i in range(128):
        sa.Table(f'config_table_{i}',metadata,
                 sa.Column('key_0',sa.Integer,primary_key=True),
                 sa.Column('key_1',sa.String,primary_key=True),
                 sa.Column('BLOCK_M',sa.Integer),sa.Column('num_warps',sa.Integer))
    metadata.create_all(engine)
    with engine.begin() as connection:
        for i in range(128):
            connection.execute(metadata.tables[f'config_table_{i}'].insert(),
                [{'key_0':1,'key_1':'bf16','BLOCK_M':16+i,'num_warps':4},
                 {'key_0':64,'key_1':'fp32','BLOCK_M':64+i,'num_warps':8}])

    def original(name):
        base=automap_base();base.prepare(engine);return base.classes.get(name)

    rows=[]
    for name in ['config_table_0','config_table_63','config_table_127','missing_table']:
        before=time.perf_counter();a=original(name);old_time=time.perf_counter()-before
        before=time.perf_counter();b=reflect_one(name,engine);new_time=time.perf_counter()-before
        assert (a is None)==(b is None)
        if a is not None:
            assert [(c.name,str(c.type),c.primary_key) for c in a.__table__.columns]==[(c.name,str(c.type),c.primary_key) for c in b.__table__.columns]
            with Session(engine) as session:
                for key in [(1,'bf16'),(64,'fp32')]:
                    x=session.get(a,key);y=session.get(b,key)
                    assert x.BLOCK_M==y.BLOCK_M and x.num_warps==y.num_warps
        rows.append(dict(table=name,missing=a is None,equivalent=True,
                         original_seconds=old_time,one_table_seconds=new_time))
    data=dict(complete=True,cpu_only=True,synthetic_database=True,
              existing_configs_modified=False,all_checks_passed=True,rows=rows,
              old_median_seconds=statistics.median(r['original_seconds'] for r in rows),
              new_median_seconds=statistics.median(r['one_table_seconds'] for r in rows))
    (root/'result.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
