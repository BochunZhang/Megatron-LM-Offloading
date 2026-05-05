import os
import json
from typing import Optional
from pycallgraph2.output import Output

class TextOutput(Output):
    def __init__(self, path:str, rank:Optional[int]=None, **kwargs):
        os.makedirs(path, exist_ok=True)
        self.filename_calls = os.path.join(path, f"calls{f'.{rank}' if rank is not None else ''}.txt")
        self.filename_nodes = os.path.join(path, f"nodes{f'.{rank}' if rank is not None else ''}.txt")
        self.filename_trace = os.path.join(path, f"trace{f'.{rank}' if rank is not None else ''}.txt")
        self.callgraph_txt  = os.path.join(path, f"callgraph{f'.{rank}' if rank is not None else ''}.txt")
        self.callgraph_json = os.path.join(path, f"callgraph{f'.{rank}' if rank is not None else ''}.json")
        Output.__init__(self, **kwargs)

    @classmethod
    def add_arguments(cls, subparsers, parent_parser, usage):
        pass

    def sanity_check(self):
        pass

    # 记录 func 被调用的次数 (只记录被 keep 的对象)
    def log_nodes(self):
        with open(self.filename_nodes, mode='w') as f:
            for i, node in enumerate(self.processor.nodes()):
                f.write(f"node[{i}]: name = {node.name}, calls = {node.calls.value}\n")

    # 记录 func 被调用的次数 (没有 keep 的对象也会被记录)
    def log_calls(self):
        with open(self.filename_calls, mode='w') as f:
            for k, v in self.processor.calls.items():
                f.write(f"name = {k}, calls = {v}\n")
                

    # 记录 trace 里面的内容
    def log_trace(self):
        with open(self.filename_trace, mode='w') as f:
            for i in range(len(self.processor.traces)):
                # call
                if len(self.processor.traces[i]) == 5:
                    t = self.processor.traces[i]
                    f.write(f"{t[0]}({type(t[0])})[{t[1]}] -> {t[2]}({type(t[2])})[{t[3]}], depth = {t[4]}\n")
                # return
                elif len(self.processor.traces[i]) == 3:
                    t = self.processor.traces[i]
                    f.write(f"{t[0]}({type(t[0])})[{t[1]}], depth = {t[2]}\n")
                    

    def log_callgraph(self):
        call_stack = [1]    # index, __main__ = 1
        names = {1: '__main__'}

        calls = {1:  0}     # 记录该函数调用了几个子函数
        calle = {1: -1}     # 记录该函数的调用者
        exits = {1: -1}     # exits == 0 表示这个节点已经结束
        nodes = {}

        main = {
            "name": "__main__",
            "indx": 1, 
            "func": []
        }
        nodes[1] = main

        max_index = -1

        # debug
        # print(call_stack)

        for t in self.processor.traces:
            # call
            if len(t) == 5:
                src, src_idx, dst, dst_idx, depth = t
                
                # some call is skipped
                if depth - 1 > len(call_stack):
                    for _ in range(depth - len(call_stack) - 1):
                        call_stack.append(-1)
                # some return is skipped
                elif depth - 1 < len(call_stack):
                    for _ in range(len(call_stack) - depth + 1):
                        assert call_stack[-1] == -1
                        call_stack.pop(-1)
                # add call 
                call_stack.append(dst_idx)

                # debug
                # print(call_stack)

                names[dst_idx] = dst            # name
                calls[dst_idx] = 0              # init
                calle[dst_idx] = -1 * src_idx if src == '' else src_idx     # caller, name == '' 意味着这是一个 non-keep 节点
                calls[src_idx] += 1             # src_func 调用了新的函数
                max_index = max(max_index, dst_idx)

                node = {
                    'name': dst,
                    'indx': dst_idx,
                    'func': []
                }
                nodes[dst_idx] = node
                nodes[src_idx]['func'].append(node)

            
            # return
            elif len(t) == 3:
                fun, fun_idx, depth = t
                if depth + 1 < len(call_stack):
                    # some return is skipped
                    for _ in range(len(call_stack) - depth - 1):
                        assert call_stack[-1] == -1
                        call_stack.pop(-1)

                idx = call_stack.pop(-1)
                exits[fun_idx] = max_index

                assert idx == fun_idx
                assert names[fun_idx] == fun
                assert depth == len(call_stack)

                # debug
                # print(call_stack)
            else:
                assert False

        with open(self.callgraph_txt, 'w') as f:
            f.write(f"{names[1]}\n")
            keys = sorted(names.keys())
            assert keys[0] == 1
            for i in range(1, len(keys)):
                for j in range(i):
                    if calle[keys[i]] < 0:
                        append = "... "
                        calle[keys[i]] = -1 * calle[keys[i]]
                    else:
                        append = ""
                    if calle[keys[i]] == keys[j]:
                        if calls[keys[j]] == 1:
                            f.write("└── ")
                        else:
                            f.write("├── ")
                        calls[keys[j]] -= 1
                    else:
                        if calls[keys[j]] == 0:
                            f.write("    ")
                        elif calls[keys[j]] > 0:
                            f.write("|   ")
                    if exits[keys[j]] == keys[i]:
                        calls[keys[j]] = -1
                if exits[keys[i]] == keys[i]:
                    calls[keys[i]] = -1
                f.write(f"{append}{names[keys[i]]}\n")

        with open(self.callgraph_json, 'w', encoding="utf-8") as j:
            print("logging...")
            print(main)
            json.dump(main, j, indent=4, ensure_ascii=False)


    def done(self):
        self.log_nodes()
        self.log_calls()
        self.log_trace()
        self.log_callgraph()
