from pyc_vm import *

class token:
    TK_NUM_INT = 1
    TK_NUM_FLOAT = 2
    TK_STRING = 3
    TK_ID = 4
    TK_IF = 5
    TK_FOR = 6
    TK_ELSE = 7
    TK_WHILE = 8
    TK_BREAK = 9
    TK_RETURN = 10
    TK_GLOBAL = 11
    TK_OTHER = 12
    
    def __init__(self):
        self.type = None
        self.value = None

class token_parser:
    def __init__(self, s):
        self.t = s
        self.t_len = len(s)
        self.p = 0
        self.done = False
        self.reserve_keys = {'if':token.TK_IF,
                             'else':token.TK_ELSE,
                            'for':token.TK_FOR,
                            'while':token.TK_WHILE,
                            'break':token.TK_BREAK,
                            'return':token.TK_RETURN,
                            'global':token.TK_GLOBAL
                            }
    def read_num(self):
        ''' d1[.d] '''
        d1 = d = ''
        t = token()
        t.type = token.TK_NUM_INT
        self.p -= 1
        
        while(self.p < self.t_len):
            c = self.t[self.p:self.p+1]
            if c == '.' and not d1:
                d1 = d
                d = ''
                t.type = token.TK_NUM_FLOAT
            elif c.isdigit():
                d += c
            else:
                break
            self.p += 1

        if t.type == token.TK_NUM_FLOAT and d:
            t.value = float(d1 + '.' + d)
        elif d:
            t.value = int(d)
        else:
            raise Exception('read_num: err')
            
        return t
    
    def read_string(self):
        t = token()
        t.type = token.TK_STRING
        p = self.t[self.p:].find('"')
        if p > 0:
            t.value = self.t[self.p:self.p + p]
            self.p += p + 1
            return t
        
        raise Exception('read_string: err')
    
    def next(self):
        t = token()
        t.type = token.TK_OTHER
        t.value = ''
        
        while(self.p < self.t_len):
            c = self.t[self.p:self.p+1]
            self.p += 1
            
            #skip space
            if c.isspace() and not t.value:
                continue
            elif c.isdigit() and not t.value:
                return self.read_num()
            elif c == '"' and not t.value:
                return self.read_string()
            elif c in ['>', '<', '=', '!', '+', '-', '*', '/'] and t.type == token.TK_OTHER:
                if (self.t[self.p:self.p+1] == '='):
                    self.p += 1
                    c += '='
                elif self.t[self.p:self.p+1] == c and c in '+-':
                    self.p += 1
                    c += c
                t.value = c
                return t
            elif c in ['&', '|'] and t.type == token.TK_OTHER:
                if self.t[self.p:self.p+1] == c:
                    self.p += 1
                    c += c
                t.value = c
                return t
            elif c.isalnum() or c == '_':#name_id
                t.type = token.TK_ID
                t.value += c
                continue
            elif t.value:
                if (t.value in self.reserve_keys.keys()):
                    t.type = self.reserve_keys[t.value]

                self.p -= 1
                return t
            
            #TK_OTHER
            t.value = c
            return t
        
        self.done = True
        return None
    
class vm_program:
    def __init__(self):        
        self.entry = 0          # main entry point
        self.op_num = 0         # total op nums
        self.globals_num = 0    # globals nums
        self.stack_num = 0       # regs nums
        self.functions = []     # index, (pc, argc, regnum, stacknum)
        self.imp_funcs = []     # index, funcname
        self.op_list = []
        
class vm_op:
    OP_ADD = 1      # op1 + op2
    OP_SUB = 2      # op1 - op2
    OP_MUL = 3      # op1 * op2
    OP_DIV = 4      # op1 / op2
    OP_PWR = 5      # op1 ^ op2
    OP_MOD = 6      # op1 % op2
    OP_EQ = 7       # op1 == op2
    OP_GT = 8       # op1 > op2
    OP_GE = 9       # op1 >= op2
    OP_LT = 10      # op1 < op2
    OP_LE = 11      # op1 <= op2
    OP_NE = 12      # op1 != op2
    OP_AND = 13     # op1 && op2
    OP_OR = 14      # op1 || op2
    OP_FUN = 15      # call op1, op2 -> argv list[]
    OP_MINUS = 16    # -right
    OP_ASSIGN = 17   # op1 <- op2
    OP_INDEX = 18   # op1[right]
    OP_FIELD = 19   # op1.right
    OP_JZ = 20      # if op1 == 0 JMP pc + op2
    OP_JMP = 21     # jmp pc + op1
    OP_PUSH = 22    # local stack
    OP_RET = 23     # function ret op2
    
    class op_num:
        OP_NUM_K = 1    # const value
        OP_NUM_R = 2    # reg temp value
        OP_NUM_S = 3    # stack argv/local value
        OP_NUM_G = 4    # global value
        def __init__(self, value = None):
            if isinstance(value, token):
                value = value.value
                
            if isinstance(value, int) \
            or isinstance(value, float) \
            or isinstance(value, basestring) :                
                self.type = self.OP_NUM_K
            else:
                self.type = None
                
            self.value = value
        def dump(self):
            type_name = ['', 'r', 'l', 'g']
            return type_name[self.type-1] + str(self.value)
    
    def __init__(self, op):
        self.op = op
        self.opr = None
        self.op1 = None
        self.op2 = None
        
    def dump(self, pc):
        op1 = op2 = opr = ''
        op_name=['+','-','*','/','^', '%', 
                     '==', '>', '>=', '<', '<=', '!=', 'AND', 'OR',
                      'call', '-', '<-', '[]', '.', 'JZ', 'JMP', 'PUSH', 'RET']
        if self.op1: op1 = self.op1.dump()
        if self.op2: op2 = self.op2.dump()
        if self.opr: opr = '->' + self.opr.dump()
        if self.op in [vm_op.OP_JZ, vm_op.OP_JMP]: op2 = int(op2)+pc
        print pc, op1, op_name[self.op - 1], op2, opr
    
class l_compiler:
    class node_info:
        def __init__(self, prev):
            self.prev = None
            self.op = None
            self.delay_ops = [] # expr | 
            
    class chunck_info:
        def __init__(self):
            self.chunck_name = ''
            self.argc = 0
            self.stacknum = 0
            self.regsnum = 0
            self.op_list = []
            
    def __init__(self, l_parser):
        self.freereg = 0
        self.l_parser = l_parser
        self.freereg_list = []
        self.op_list = []
        self.local_vars = {}    # key->var_name value->index
        self.global_vars = {}   # key->var_name value->index
        self.chucks_dict = {}
        
    def new_reg(self):
        if self.freereg_list:
            return self.freereg_list.pop()
        
        self.freereg += 1
        return self.freereg - 1
    
    def new_global_var(self, var_name):
        if var_name in self.local_vars.keys() \
        or var_name in self.global_vars.keys():
            raise Exception("global_var conflict:", var_name)
        
        num = vm_op.op_num(len(self.global_vars))
        num.type = vm_op.op_num.OP_NUM_G
        self.global_vars[var_name] = num
        return num
    
    def get_var(self, var_name):
        if var_name in self.global_vars.keys():
            return self.global_vars[var_name]
        
        if not var_name in self.local_vars.keys():
            num = vm_op.op_num(len(self.local_vars))
            num.type = vm_op.op_num.OP_NUM_S
            self.local_vars[var_name] = num
            
        return self.local_vars[var_name]
    
    def free_reg(self, reg_id):
        self.freereg_list.append(reg_id)
        
    def free_expr_regs(self, node):
        #free expr regs
        if isinstance(node, exp_op) and node.ret_id != None:
            self.free_reg(node.ret_id)
        
        #free argv list regs
        if isinstance(node, list):
            for expr in node:
                self.free_expr_regs(expr);
                
    def compile_whilestmt(self, whilestmt, node):
        expr = whilestmt.expr
        
        pc = len(self.op_list) - 1
        
        jz_op = vm_op(vm_op.OP_JZ)
        jz_op.op1 = self.do_compile(expr, node)
        self.op_list.append(jz_op)        
        
        self.free_expr_regs(expr)
        
        for chunck_item in whilestmt.chunck:
            self.do_compile(chunck_item, node)
        
        jmp_op = vm_op(vm_op.OP_JMP)
        jmp_op.op2 = vm_op.op_num(pc - (len(self.op_list) - 1))
        self.op_list.append(jmp_op)
        
        # patch jz/jmp op2 -> jump out
        for i in xrange(len(self.op_list) - pc):
            if self.op_list[pc+i].op in [vm_op.OP_JZ, vm_op.OP_JMP] \
            and self.op_list[pc+i].op2 is None:
                self.op_list[pc+i].op2 = vm_op.op_num(len(self.op_list) - (pc + i))
        
    def compile_ifstmt(self, ifstmt, node):
        jmp_list = []
        last_item = ifstmt.if_chuncks[-1]
        
        for if_item in ifstmt.if_chuncks:
            expr, chunck = if_item
            if expr:
                # INSERT JZ NEXT PC
                jz_op = vm_op(vm_op.OP_JZ)
                jz_op.op1 = self.do_compile(expr, node)
                self.op_list.append(jz_op)
                
                self.free_expr_regs(expr)
                
            pc = len(self.op_list) - 1
            for chunck_item in chunck:
                self.do_compile(chunck_item, node)
            
            if if_item != last_item:
                # INSET JMP OUT
                self.op_list.append(vm_op(vm_op.OP_JMP))
                jmp_list.append(len(self.op_list) - 1)
            
            self.op_list[pc].op2 = vm_op.op_num(len(self.op_list) - pc)
        
        for jmp in jmp_list:
            self.op_list[jmp].op2 = vm_op.op_num(len(self.op_list) - jmp)

    def comile_forstmt(self, forstmt, node):
        for stmt in forstmt.init_chuncks:
            self.do_compile(stmt, node)
        
        pc = len(self.op_list) - 1
        if forstmt.expr:
            jz_op = vm_op(vm_op.OP_JZ)
            jz_op.op1 = self.do_compile(forstmt.expr, node)
            self.op_list.append(jz_op)
            
        for stmt in forstmt.chunck:
            self.do_compile(stmt, node)
        
        for stmt in forstmt.inc_chuncks:
            self.do_compile(stmt, node)
            
        jmp_op = vm_op(vm_op.OP_JMP)
        jmp_op.op2 = vm_op.op_num(pc - (len(self.op_list) - 1))
        self.op_list.append(jmp_op)

        # patch jz/jmp op2 -> jump out
        for i in xrange(len(self.op_list) - pc):
            if self.op_list[pc+i].op in [vm_op.OP_JZ, vm_op.OP_JMP] \
            and self.op_list[pc+i].op2 is None:
                self.op_list[pc+i].op2 = vm_op.op_num(len(self.op_list) - (pc + i))
    
    def do_delay_expr(self, expr):        
        if expr.delay_op == exp_op.EXPR_OP_FLAGS_INCINC:
            r_expr = exp_op(vm_op.OP_ADD)
        elif expr.delay_op == exp_op.EXPR_OP_FLAGS_DECDEC:
            r_expr = exp_op(vm_op.OP_SUB)
        else:
            raise Exception('delay_expr unkonwn op')
        
        expr.delay_op = None
        # expr = expr + 1
        token_1 = token()
        token_1.type = token.TK_NUM_INT
        token_1.value = 1
            
        r_expr.left = expr
        r_expr.right = exp_value(token_1)
            
        assign_expr = exp_op(vm_op.OP_ASSIGN)
        assign_expr.left = expr
        assign_expr.right = r_expr

        self.compile_expr(assign_expr, None)
        
    def test_delay_expr(self, cur_node):
        if not isinstance(cur_node.op, exp_op) \
        and not isinstance(cur_node.op, exp_value): return
        
        if cur_node.op.delay_op:
            # get the last exp_op node
            item = cur_node.prev
            last_item = None
            while (item and isinstance(item.op, exp_op)):
                last_item = item
                item = item.prev
                
            if last_item:
                last_item.delay_ops.append(cur_node.op)
            else:
                self.do_delay_expr(cur_node.op)
        elif len(cur_node.delay_ops) > 0:
            for item in cur_node.delay_ops:
                self.do_delay_expr(item)
            cur_node.delay_ops = []
            
    def compile_expr(self, expr, node):
        op = vm_op(expr.op)
        op.op1 = self.do_compile(expr.left, node)
        
        if isinstance(expr.right, list):
            # return argv num
            for item in reversed(expr.right):
                push_op = vm_op(vm_op.OP_PUSH)
                push_op.op2 = self.do_compile(item, node)
                self.op_list.append(push_op)
            op.op2 = vm_op.op_num(len(expr.right))
        else:
            op.op2 = self.do_compile(expr.right, node)
        
        self.free_expr_regs(expr.left)
        self.free_expr_regs(expr.right)

        self.op_list.append(op)
        
        # test node delay_expr
        if node: self.test_delay_expr(node)
        
        # OP_ASSIGN no need new_reg
        if expr.op == vm_op.OP_ASSIGN:            
            return op.op1
        
        # function no need new_reg
        if expr.op == vm_op.OP_FUN:
            if not node.prev or not isinstance(node.prev.op, exp_op):
                return None
        
        # allocate freereg -> ret
        expr.ret_id = self.new_reg()
        op.opr = vm_op.op_num(expr.ret_id)
        op.opr.type = vm_op.op_num.OP_NUM_R
        
        return op.opr
    
    def get_exp_value(self, cur_node):
        if cur_node.op.value.type != token.TK_ID:
            return vm_op.op_num(cur_node.op.value)
        
        if not cur_node.prev or not isinstance(cur_node.prev.op, exp_op):
            return self.get_var(cur_node.op.value.value)
        
        # function name or field name ?
        if cur_node.prev.op.op == vm_op.OP_FUN \
        and cur_node.prev.op.left == cur_node.op:
            return vm_op.op_num(cur_node.op.value)
        elif cur_node.prev.op.op == vm_op.OP_FIELD \
        and cur_node.prev.op.right == cur_node.op:
            return vm_op.op_num(cur_node.op.value)
        
        return self.get_var(cur_node.op.value.value)

    def do_compile(self, item, prev_node = None):
        cur_node = self.node_info(prev_node)
        cur_node.prev = prev_node
        cur_node.op = item
        
        self.test_delay_expr(cur_node)
        
        if isinstance(item, exp_op):
            return self.compile_expr(item, cur_node)
        elif isinstance(item, exp_value):
            return self.get_exp_value(cur_node)
        elif isinstance(item, if_state):
            self.compile_ifstmt(item, cur_node)
        elif isinstance(item, while_state):
            self.compile_whilestmt(item, cur_node)
        elif isinstance(item, for_state):
            self.comile_forstmt(item, cur_node)
        elif isinstance(item, global_var_state):
            assert (not prev_node)
            for var in item.vars:
                self.new_global_var(var.value)
        elif isinstance(item, state_key_word):
            t, v = item.value
            if t == token.TK_BREAK:
                self.op_list.append(vm_op(vm_op.OP_JMP))    #JMP OUT
            elif t == token.TK_RETURN:
                ret_op = vm_op(vm_op.OP_RET)
                if v: ret_op.op2 = self.do_compile(v, cur_node)
                self.op_list.append(ret_op)
    
    def dump_op_list(self):
        pc = 0
        for op in self.op_list:
            op.dump(pc)
            pc += 1
    
    def compile_func(self, funbody, name = None):
        chunck = self.chunck_info()
        chunck.argc = len(funbody.argvs)
        if name:
            chunck.chunck_name = name
        else:
            chunck.chunck_name = funbody.funname.value
        
        if chunck.chunck_name in self.chucks_dict.keys():
            raise Exception("compile_func confict:", chunck.chunck_name)
        
        self.op_list = []
        self.local_vars = {}
        self.freereg = 0
        self.freereg_list = []
        
        # init function call argvs
        for argv in funbody.argvs:
            num = vm_op.op_num(len(self.local_vars))
            num.type = vm_op.op_num.OP_NUM_S
            num.value = -(len(self.local_vars) + 1)
            self.local_vars[argv.value] = num
        
        for op in funbody.chunck:
            self.do_compile(op)
        
        chunck.op_list = self.op_list
        chunck.op_list.append(vm_op(vm_op.OP_RET))
        chunck.stacknum = len(self.local_vars)
        chunck.regsnum = self.freereg + len(self.freereg_list)
        
        self.chucks_dict[chunck.chunck_name] = chunck
        
        return chunck
    
    def run(self):
        # main chunck
        main_func = funbody_state()
        for item in self.l_parser.parser_list:
            if isinstance(item, funbody_state):
                self.compile_func(item)
            else:
                main_func.chunck.append(item)
                
        # compile functions
        self.compile_func(main_func, 'main')
    
    def run_linker(self):
        # do linker -> vm_program
        functions = {}
        imp_funcs = {}
                
        program = vm_program()
        program.globals_num = len(self.global_vars)
        
        for func in self.chucks_dict.iterkeys():
            chuck = self.chucks_dict[func]            
            program.functions.append((len(program.op_list), chuck.argc, chuck.stacknum, chuck.regsnum))
            functions[func] = (len(program.functions) - 1, program.functions[-1])
            program.op_list += chuck.op_list
            
        # fix fun call
        for op in program.op_list:
            if op.op != vm_op.OP_FUN:
                continue
            
            assert isinstance(op.op1, vm_op.op_num)
            if op.op1.type != vm_op.op_num.OP_NUM_K:
                continue
            
            fun_name = op.op1.value
            if fun_name in functions.keys():
                # op1 -> local function pc
                fun_id, fun_info = functions[fun_name]
                op.op1 = vm_op.op_num(fun_id)
                if op.op2.value != fun_info[1]:
                    raise Exception(fun_name, "argc num error")
            elif fun_name in imp_funcs.keys():
                op.op1 = imp_funcs[fun_name]
            else:
                imp_funcs[fun_name] = vm_op.op_num(-(len(program.imp_funcs) + 1))
                program.imp_funcs.append(fun_name)
                op.op1 = imp_funcs[fun_name]
         
        # setup entry -> main
        entry_op = vm_op(vm_op.OP_FUN)
        entry_op.op1 = vm_op.op_num(functions['main'][0])
        entry_op.op2 = vm_op.op_num(0)
        program.op_list.append(entry_op)
        program.op_list.append(vm_op(vm_op.OP_RET))
        #program.op_list
        program.entry = len(program.op_list) - 2
        
        return program
'''
l_parser node define -> 
exp_value | exp_op | if_state | while_state | for_state
'''
class state_key_word:
    def __init__(self, vaule):
        self.value = vaule
        
class exp_value:
    def __init__(self, vaule):
        self.value = vaule
        self.delay_op = None
        
class exp_op:
    EXPR_OP_FLAGS_INCINC = 1
    EXPR_OP_FLAGS_DECDEC = 2
    '''l_parser_expr_node'''
    def __init__(self, op):
        self.op = op
        self.left = None
        self.right = None
        self.ret_id = None      # result reg_index
        self.delay_op = None
        
class if_state:
    def __init__(self):
        '''if_chunck -> [(expr1, chunck1), ... ]
        '''
        self.if_chuncks = []
        
class while_state:
    def __init__(self):
        self.expr = None
        self.chunck = None

class for_state:
    def __init__(self):
        self.init_chuncks = []
        self.expr = None
        self.inc_chuncks = []
        self.chunck = None

class funbody_state:
    def __init__(self):
        self.funname = None
        self.argvs = []
        self.chunck = []
        
class global_var_state:
    def __init__(self):
        self.vars = []
        self.chunck = []
        
class l_parser:
    UNARY_PRIORITY = 8
    def __init__(self, tk_parser):
        self.tk = tk_parser
        self.tk_list = []
        self.tk_index = -1
        self.parser_list = []   # l_parser result tree
        self.op_info = {
                '+':(vm_op.OP_ADD, 6, 6),
                '-':(vm_op.OP_SUB, 6, 6),
                '*':(vm_op.OP_MUL, 7, 7),
                '/':(vm_op.OP_DIV, 7, 7),
                '%':(vm_op.OP_MOD, 7, 7),
                '^':(vm_op.OP_PWR, 10, 9),
                '>':(vm_op.OP_GT, 3, 3),
                '<':(vm_op.OP_LT, 3, 3),
                '>=':(vm_op.OP_GE, 3, 3),
                '==':(vm_op.OP_EQ, 3, 3),
                '!=':(vm_op.OP_NE, 3, 3),
                '<=':(vm_op.OP_LE, 3, 3),
                '&&':(vm_op.OP_AND, 2, 2),
                '||':(vm_op.OP_OR, 2, 2),
                }
        
        self.last_pattern = None
        self.last_tk = None
        
    def get_next_tk(self):
        if self.tk_index < len(self.tk_list) - 1:            
            self.tk_index += 1
            return self.tk_list[self.tk_index]
        
        nxt_tk = self.tk.next()
        if nxt_tk is None: return None
        
        self.tk_list.append(nxt_tk)
        self.tk_index = len(self.tk_list) - 1
        return self.tk_list[-1]
    
    def test_next_tk(self, c):
        tk = self.get_next_tk()
        if tk and tk.value != c:
            self.tk_index -= 1
            return None
        self.last_tk = tk
        return tk
    
    def test_next_tk_type(self, t):
        tk = self.get_next_tk()
        if tk and tk.type != t:
            self.tk_index -= 1
            return None
        self.last_tk = tk
        return tk
    
    def assignment(self):
        ''' assignment->
        primaryexp '=' primaryexp | expr | assignment
        '''
        assignment_dict = {'=':vm_op.OP_ASSIGN, 
                           '+=':vm_op.OP_ADD, 
                           '-=':vm_op.OP_SUB, 
                           '*=':vm_op.OP_MUL, 
                           '/=':vm_op.OP_DIV}
        l_expr = self.test_pattern(self.primaryexp)
        if l_expr:
            tk = self.get_next_tk()
            if tk.value in assignment_dict.keys():
                if self.test_pattern(self.assignment) or self.test_pattern(self.expr):
                    assign = exp_op(vm_op.OP_ASSIGN)
                    assign.left = l_expr                    
                    if tk.value != '=':
                        op = exp_op(assignment_dict[tk.value])
                        op.left = l_expr
                        op.right = self.last_pattern
                        assign.right = op
                    else:
                        assign.right = self.last_pattern
                        
                    return assign
            else:
                self.tk_index -= 1
                
        return None
    
    def test_pattern(self, pattern):
        tk_index = self.tk_index
        result = pattern()
        if not result:
            self.tk_index = tk_index
        else:
            self.last_pattern = result
        return result
        
    def simpleexp(self):
        ''' NUMBER | STRING | primaryexp
        '''
        t = self.get_next_tk()
        if t.type in [token.TK_STRING, token.TK_NUM_INT, token.TK_NUM_FLOAT]:
            return exp_value(t)
            
        self.tk_index -= 1
        return self.test_pattern(self.primaryexp)
    
    def test_expr_op(self):
        c = self.get_next_tk()
        
        if c.value in self.op_info.keys():
            return self.op_info[c.value]
    
        self.tk_index -= 1
        return None
    
    def expr(self, priority = 0):
        '''
        [+-](simpleexp) { op expr }
        expr -> (simpleexp | unop expr) { binop expr }
        ''' 
        tk = self.get_next_tk()
        l_exp = r_exp = None
        
        # first test unary-expression <unary-operator> '+ | -'
        if tk.value == '+' or tk.value == '-':
            r_exp = self.expr(self.UNARY_PRIORITY)
            assert (r_exp)
            if tk.value == '-':
                minus_exp = exp_op(vm_op.OP_MINUS)
                minus_exp.right = r_exp
                l_exp = minus_exp
        else:
            self.tk_index -= 1
            #l_exp = simpleexp
            l_exp = self.test_pattern(self.simpleexp)
            
        while(l_exp):
            op_info = self.test_expr_op()
            if not op_info:
                break
            
            op_type, l_priority, r_priority = op_info
            if priority >= l_priority:
                self.tk_index -= 1
                return l_exp
                    
            r_exp = self.expr(r_priority)
            assert(r_exp)
                
            parent_exp = exp_op(op_type)
                
            #l_exp = l_exp op r_exp
            
            parent_exp.left = l_exp
            parent_exp.right = r_exp
            #parent_exp
            l_exp = parent_exp
            
        return l_exp
    
    def explist(self, parent=None):
        '''explist->
        assignment | expr { `,' explist }
        '''
        if not self.test_pattern(self.assignment) and \
            not self.test_pattern(self.expr):
            return []
        
        expr_list = [self.last_pattern]
        while (self.test_next_tk(',')):
            if self.test_pattern(self.assignment) or self.test_pattern(self.expr):
                expr_list.append(self.last_pattern)
            else:
                break
                
        return expr_list
    
    def prefixexp(self):
        '''prefixexp -> NAME | '(' expr ')' */
        '''
        tk = self.test_next_tk_type(token.TK_ID)  
        if tk:
            return exp_value(tk)
        
        if self.test_next_tk('('):
            expr = self.test_pattern(self.expr)
            if expr and self.test_next_tk(')'):
                return expr
            
            raise Exception("prefixexp ( error")
        
        return None
    
    def primaryexp(self):
        '''primaryexp ->
        prefixexp  { '.' fieldname | '[' expr ']' | '(' func explist ')'} 
        eg: func(1, 2) | data [2+3].field1[1].field[2]
        '''
        l_expr = self.test_pattern(self.prefixexp)
        
        while (l_expr):
            if self.test_next_tk('('):
                #function
                fun_exp = exp_op(vm_op.OP_FUN)
                argv_list = self.explist(fun_exp)
                if (self.test_next_tk(')')):
                    fun_exp.left = l_expr
                    fun_exp.right = argv_list
                    l_expr = fun_exp
                else:
                    raise Exception("primaryexp fuction error")
            elif self.test_next_tk('.'):
                field_name = self.test_next_tk_type(token.TK_ID)
                if field_name:
                    field_exp = exp_op(vm_op.OP_FIELD)
                    field_exp.left = l_expr
                    field_exp.right = exp_value(field_name)
                    l_expr = field_exp
                else:
                    raise Exception("primaryexp fieldname error")
            elif self.test_next_tk('['):
                r_expr = self.test_pattern(self.expr)
                if r_expr and self.test_next_tk(']'):
                    index_expr = exp_op(vm_op.OP_INDEX)
                    index_expr.left = l_expr
                    index_expr.right = r_expr
                    l_expr = index_expr
                else:
                    raise Exception("primaryexp index error")
            elif self.test_next_tk('++'):
                l_expr.delay_op = exp_op.EXPR_OP_FLAGS_INCINC
                break
            elif self.test_next_tk('--'):
                l_expr.delay_op = exp_op.EXPR_OP_FLAGS_DECDEC
                break
            else:
                break
            
        return l_expr
    
    def forstat(self):
        '''
        for '(' init_explist; expr; inc_explist ')' {block}
        '''
        for_stmt = for_state()
        if (self.test_next_tk_type(token.TK_FOR)):
            if not self.test_next_tk('('):
                raise Exception("forstat without (")
            
            for_stmt.init_chuncks = self.explist(for_stmt)
            
            if not self.test_next_tk(';'):
                raise Exception("forstat init_chunck without ;")
            if not self.test_pattern(self.expr):
                raise Exception("forstat expr error")
            if not self.test_next_tk(';'):
                raise Exception("forstat expr without ;")
            
            for_stmt.expr = self.last_pattern
            for_stmt.inc_chuncks = self.explist(for_stmt)
            
            if not self.test_next_tk(')'):
                raise Exception("forstat without )")
            
            for_stmt.chunck = self.block()
            return for_stmt
        
        return None
            
    def whilestat(self):
        '''
        while '(' <expr> ')' {block}
        '''
        while_stmt = while_state()
        if (self.test_next_tk_type(token.TK_WHILE)):
            if not self.test_next_tk('('):
                raise Exception("whilestat without (")
            if not self.test_pattern(self.expr):
                raise Exception("whilestat expr error")
            if not self.test_next_tk(')'):
                raise Exception("whilestat expr without )")
            
            while_stmt.expr = self.last_pattern
            while_stmt.chunck = self.block()
            return while_stmt
    
        return None
    
    def ifstat(self):
        '''ifstat->
        if '(' <expr> ')' {block} {else if | else {block}}
        '''
        if_stmt = if_state()
        while (self.test_next_tk_type(token.TK_IF)):
            if not self.test_next_tk('('):
                raise Exception("ifstat without (")
            if not self.test_pattern(self.expr):
                raise Exception("ifstat expr error")
            if not self.test_next_tk(')'):
                raise Exception("ifstat expr without )")
            
            expr = self.last_pattern            
            if_stmt.if_chuncks.append((expr, self.block()))
            
            if self.test_next_tk_type(token.TK_ELSE):
                if self.test_next_tk_type(token.TK_IF):
                    self.tk_index -= 1
                    continue
                
                if_stmt.if_chuncks.append((None, self.block()))
                break
        
        if len(if_stmt.if_chuncks) <= 0:
            return None
        
        return if_stmt
    
    def funbody(self):
        '''
        funcname '(' {arg ',' } ')' {blocks}
        '''
        fun_def = funbody_state()
        fun_def.funname = self.test_next_tk_type(token.TK_ID)
        if not fun_def.funname or not self.test_next_tk('('):
            return None
        
        while (self.test_next_tk_type(token.TK_ID)):
            fun_def.argvs.append(self.last_tk)
            if not self.test_next_tk(','):
                break
        
        if not self.test_next_tk(')'):
            return None
        
        if not self.test_next_tk('{'):
            return None
        
        self.tk_index -= 1
        fun_def.chunck = self.test_pattern(self.block)
        return fun_def
    
    def statement(self):
        '''statement ->
            assignment | function | if | ...
        '''
        if self.test_pattern(self.assignment):
            return self.last_pattern
        elif self.test_pattern(self.ifstat):
            return self.last_pattern
        elif self.test_pattern(self.whilestat):
            return self.last_pattern
        elif self.test_pattern(self.forstat):
            return self.last_pattern
        elif self.test_pattern(self.primaryexp):
            # ++ / --
            if self.last_pattern and self.last_pattern.delay_op:
                return self.last_pattern
            
            # function call
            if not isinstance(self.last_pattern, exp_op) \
            or self.last_pattern.op != vm_op.OP_FUN:
                raise Exception("primaryexp in statement no function")
            
            return self.last_pattern
        
        return False
    
    def global_vars_def(self):
        '''
        'global' varname | varname = expr, ;
        '''
        var_defs = global_var_state()
        if not self.test_next_tk_type(token.TK_GLOBAL):
            return None
        
        while (self.test_next_tk_type(token.TK_ID)):
            var_defs.vars.append(self.last_tk)
            if self.test_next_tk('=') and self.test_pattern(self.expr):
                var_defs.chunck.append(self.last_pattern)
            
            if not self.test_next_tk(','):
                break
        
        if not self.test_next_tk(';'):
            raise Exception("global_vars_def end without ;")
        
        return var_defs
    
    def block(self):
        '''block -> 
        { '{' statement [`;'] '}' }
        '''
        block_start = self.test_next_tk('{')
        blocks = []
        while (not self.tk.done):
            if (self.test_pattern(self.global_vars_def)):
                blocks.append(self.last_pattern)
            elif (self.test_pattern(self.funbody)):
                blocks.append(self.last_pattern)
            elif (self.test_pattern(self.statement)):
                last_tk = self.tk_list[self.tk_index]                
                if last_tk.value != ';' and last_tk.value != '}':
                    if not self.test_next_tk(';'):
                        raise Exception("statement without ;")
                blocks.append(self.last_pattern)            
            elif (self.test_next_tk_type(token.TK_BREAK)):
                if not self.test_next_tk(';'):
                    raise Exception("break without ;")
                blocks.append(state_key_word((token.TK_BREAK, None)))
            elif (self.test_next_tk_type(token.TK_RETURN)):
                expr = self.test_pattern(self.expr)
                if not self.test_next_tk(';'):
                    raise Exception("return without ;")
                blocks.append(state_key_word((token.TK_RETURN, expr)))
            elif self.test_next_tk(';'):
                pass
            elif block_start and self.test_next_tk('}'):
                return blocks
            else:
                raise Exception("block test error")
            
            if not block_start:
                break
                    
        return blocks
        
    def run(self):
        while (not self.tk.done):
            block = self.block()
            if not block: break
            self.parser_list += block
        
if __name__ == '__main__':    
    tk = token_parser('''
    create_list(num){
        l = list();
        for(i=0; i<num;i++){
            if (len(l) > 10) break;
            l.append(i);
            }
        return l;
    }
    a = b = 4;
    c = create_list(1+a*5);
    if (a == 0) a+= 1;    
    b += -1;
    printf("%d %d %d\n", a, b, len(c));
    '''
    )
    
    parser = l_parser(tk)
    parser.run()
    
    compiler = l_compiler(parser)
    compiler.run()
    
    compiler.dump_op_list()
    
    program = compiler.run_linker()
    
    vm = py2c_vm()
    vm.run(program)