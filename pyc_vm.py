from main import vm_op, vm_program

class py2c_vm:
    MAX_STACK_NUM = 1024
    class call_info:
        def __init__(self):
            self.ret_pc = 0
            self.ret = None
            self.stacks_num = 0
            self.regs_num = 0
            self.base_stack = 0
            self.argc = 0
            
    def __init__(self):
        self.pc = 0
        self.cur_stack = 0
        self.stacks = []
        self.globals = []
        self.call_infos = []
        
    def get_opnum(self, opnum):
        assert (len(self.call_infos) > 0)
        ci = self.call_infos[-1]
        if opnum.type == vm_op.op_num.OP_NUM_K:
            return opnum.value
        elif opnum.type == vm_op.op_num.OP_NUM_R:
            return self.stacks[ci.base_stack - opnum.value]
        elif opnum.type == vm_op.op_num.OP_NUM_S:
            if opnum.value < 0: #function argv
                return self.stacks[ci.base_stack - opnum.value]
            else:
                return self.stacks[ci.base_stack - opnum.value - ci.regs_num]
        elif opnum.type == vm_op.op_num.OP_NUM_G:
            return self.globals[opnum.value]
        
    def set_opnum(self, opnum, value):
        assert (len(self.call_infos) > 0)
        assert (opnum.type != vm_op.op_num.OP_NUM_K)
        ci = self.call_infos[-1]
        if opnum.type == vm_op.op_num.OP_NUM_R:
            self.stacks[ci.base_stack - opnum.value] = value
        elif opnum.type == vm_op.op_num.OP_NUM_S:
            if opnum.value < 0: #function argv
                self.stacks[ci.base_stack - opnum.value] = value
            else:
                self.stacks[ci.base_stack - opnum.value - ci.regs_num] = value
        elif opnum.type == vm_op.op_num.OP_NUM_G:
            self.globals[opnum.value] = value
    
    def imp_printf(self, argvs):
        fmt = argvs[0]
        argvs = tuple(argvs[1:])
        print fmt % argvs,
    
    def imp_list(self, argvs):
        return list()
    
    def imp_len(self, argvs):
        assert len(argvs) == 1
        return len(argvs[0])
    
    def imp_func(self, imp_name, argvs):
        imp_lib_func = {'printf':self.imp_printf,
                        'list':self.imp_list,
                        'len':self.imp_len}
        return imp_lib_func[imp_name](argvs)
    
    def run(self, program):
        self.pc = program.entry
        self.cur_stack = self.MAX_STACK_NUM - 1
        self.stacks = [None for i in range(self.MAX_STACK_NUM)]
        self.globals = [None for i in xrange(program.globals_num)]
        
        while(True):
            op = program.op_list[self.pc]
            if op.op == vm_op.OP_FUN:
                ci = self.call_info()
                fun_id = op.op1.value
                argc = op.op2.value
                if op.op1.type != vm_op.op_num.OP_NUM_K:
                    # dynamic call function object
                    fun_obj = self.get_opnum(op.op1)
                    argvs = [self.stacks[i+self.cur_stack+1] for i in xrange(argc)]
                    #print fun_obj, op.op1.value, op.op1.type, argc
                    apply(fun_obj, argvs)
                    self.pc += 1
                    continue
                
                if fun_id < 0:
                    imp_name = program.imp_funcs[-fun_id-1]
                    argvs = [self.stacks[i+self.cur_stack+1] for i in xrange(argc)]
                    #print imp_name, argvs
                    result = self.imp_func(imp_name, argvs)
                    if not op.opr is None:
                        self.set_opnum(op.opr, result)
                    self.pc += 1
                    continue
                
                pc, ci.argc, ci.stacks_num, ci.regs_num = program.functions[fun_id]

                ci.base_stack = self.cur_stack
                self.cur_stack -= ci.stacks_num + ci.regs_num
                
                ci.ret_pc = self.pc + 1
                ci.ret = op.opr
                
                self.call_infos.append(ci)
                self.pc = pc
                continue
            elif op.op == vm_op.OP_RET:
                if op.op2 != None: result = self.get_opnum(op.op2)
                if len(self.call_infos) <= 0:
                    break
                
                ci = self.call_infos.pop()
                if op.op2 != None and ci.ret != None:
                    self.set_opnum(ci.ret, result)
                    
                self.cur_stack += ci.stacks_num + ci.regs_num + ci.argc
                self.pc = ci.ret_pc
                continue
            elif op.op == vm_op.OP_JZ:
                if (not self.get_opnum(op.op1)):
                    self.pc += self.get_opnum(op.op2)
                    continue
            elif op.op == vm_op.OP_JMP:
                self.pc += self.get_opnum(op.op2)
                continue
            elif op.op == vm_op.OP_PUSH:
                self.stacks[self.cur_stack] = self.get_opnum(op.op2)
                self.cur_stack -= 1
            elif op.op == vm_op.OP_ADD:
                self.set_opnum(op.opr, self.get_opnum(op.op1) + self.get_opnum(op.op2))
            elif op.op == vm_op.OP_SUB:
                self.set_opnum(op.opr, self.get_opnum(op.op1) - self.get_opnum(op.op2))
            elif op.op == vm_op.OP_MUL:
                self.set_opnum(op.opr, self.get_opnum(op.op1) * self.get_opnum(op.op2))
            elif op.op == vm_op.OP_DIV:
                self.set_opnum(op.opr, self.get_opnum(op.op1) / self.get_opnum(op.op2))
            elif op.op == vm_op.OP_PWR:
                self.set_opnum(op.opr, self.get_opnum(op.op1) ^ self.get_opnum(op.op2))
            elif op.op == vm_op.OP_MOD:
                self.set_opnum(op.opr, self.get_opnum(op.op1) % self.get_opnum(op.op2))
            elif op.op == vm_op.OP_ASSIGN:
                self.set_opnum(op.op1, self.get_opnum(op.op2))
            elif op.op == vm_op.OP_LE:
                self.set_opnum(op.opr, self.get_opnum(op.op1) <= self.get_opnum(op.op2))
            elif op.op == vm_op.OP_LT:
                self.set_opnum(op.opr, self.get_opnum(op.op1) < self.get_opnum(op.op2))
            elif op.op == vm_op.OP_GT:
                self.set_opnum(op.opr, self.get_opnum(op.op1) > self.get_opnum(op.op2))
            elif op.op == vm_op.OP_GE:
                self.set_opnum(op.opr, self.get_opnum(op.op1) >= self.get_opnum(op.op2))
            elif op.op == vm_op.OP_EQ:
                self.set_opnum(op.opr, self.get_opnum(op.op1) == self.get_opnum(op.op2))
            elif op.op == vm_op.OP_NE:
                self.set_opnum(op.opr, self.get_opnum(op.op1) != self.get_opnum(op.op2))
            elif op.op == vm_op.OP_AND:
                self.set_opnum(op.opr, self.get_opnum(op.op1) and self.get_opnum(op.op2))
            elif op.op == vm_op.OP_OR:
                self.set_opnum(op.opr, self.get_opnum(op.op1) or self.get_opnum(op.op2))
            elif op.op == vm_op.OP_MINUS:
                self.set_opnum(op.opr, -self.get_opnum(op.op2))
            elif op.op == vm_op.OP_FIELD:
                x = getattr(self.get_opnum(op.op1), self.get_opnum(op.op2))
                self.set_opnum(op.opr, x)
            
            self.pc += 1