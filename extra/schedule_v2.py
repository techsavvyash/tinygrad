from typing import List, Optional, Set, TypeVar, Callable
from tinygrad import Tensor
from tinygrad.realize import create_schedule
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import ScheduleItem
from graphlib import TopologicalSorter

# this class should be usable for both UOps and LazyBuffers
T = TypeVar("T")
class DAG:
  def __init__(self, sinks:List[T], get_parents:Callable[[T], List[T]]):
    self.ts = TopologicalSorter()
    remaining = sinks[:]
    seen = set()
    while len(remaining):
      n = remaining.pop()
      if n in seen: continue
      seen.add(n)
      remaining += (parents := get_parents(n))
      self.ts.add(n, *parents)
  def print_static_order(self):
    for x in self.ts.static_order():
      print(id(x), x)
def create_schedule_v2(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  # toposort outs and their deps
  # is the shapetracker on the edge?
  #dag = DAG([x.base for x in outs], lambda x: [x.base for x in x.srcs] if x.base.realized is None else [])
  dag = DAG(outs, lambda x: (x.srcs if x.realized is None else []) if x is x.base else [x.base])
  dag.print_static_order()
  pass

if __name__ == "__main__":
  x = Tensor.rand(32,32).realize()
  w1 = Tensor.rand(32,32).realize()
  w2 = Tensor.rand(32,32).realize()

  x = x @ w1
  x = x @ w2

  sched = create_schedule_v2([x.lazydata])

  #sched = create_schedule([x.lazydata])

  #for s in sched: print(s)

  #x.realize()