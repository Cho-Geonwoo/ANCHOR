# Source lineage and third-party notices

The ANCHOR implementation and baseline update machinery are derived from the author's
WSRL research fork, commit `230a88a` ("Add matched PEX policy expansion baseline").
The upstream implementation is [zhouzypaul/wsrl](https://github.com/zhouzypaul/wsrl).
The `wsrl` namespace is retained to make this lineage visible.

WSRL credits [dibyaghosh/jaxrl_minimal](https://github.com/dibyaghosh/jaxrl_minimal),
including contributions from Kevin Black, Homer Walke, Kyle Stachowicz, and others.
The Adroit dataset loader records its source in the Cal-QL implementation:
<https://github.com/nakamotoo/Cal-QL/blob/ac6eafec22e8d60836573e1f488c7f626ce8a77e/JaxCQL/replay_buffer.py>.

The provided research checkout had an MIT badge in its README but did not contain
a repository LICENSE file. The inspected upstream WSRL tree also did not contain
a LICENSE file. No repository-wide license is assigned by this cleanup; the intended
license and permission to redistribute the inherited WSRL code need to be established
with its authors before a public release. A badge is not a license grant.

The installer retrieves these dependencies separately and preserves their own notices:

| Dependency | Repository | Pinned commit |
| --- | --- | --- |
| D4RL with WSRL Kitchen reward fix | https://github.com/zhouzypaul/D4RL | d005dca9dab0c0c50aa5aa155cdd6b64de1cfe4d |
| Adroit binary environments | https://github.com/nakamotoo/mj_envs | 88fadb11c38e8141dfe7bbde92a8954858c4b9f2 |
| mjrl | https://github.com/aravindr93/mjrl | 3871d93763d3b49c4741e6daeaebbc605fe140dc |

The public Adroit demonstration archive remains an external dataset. The manuscript
summary table in `data/` is included to identify reported reference values; it is not
training data or raw experiment measurements.
