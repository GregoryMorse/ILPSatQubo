#pip install numpy
#5 solvers used for substitution searching (scipy is used if substitution functions are known, or simple case detected)
#optional: pip install scipy
#optional: pip install gurobipy #must install Gurobi (such as the free or academic version)
#optional: pip install cplex #must install IBM ILOG CPLEX Optimization Studio (such as the free or academic version)
#optional: pip install z3-solver
#optional: pip install glpk
#for quantum annealer:
#pip install dimod
#pip install minorminer
#pip install dwave-system

NOT, ALWAYSFALSE, ALWAYSTRUE, ALWAYSLEFT, ALWAYSNOTLEFT, ALWAYSRIGHT, ALWAYSNOTRIGHT = 0, 1, 2, 3, 4, 5, 6 #trivial boolean operators
AND, NAND, OR, NOR, NIMPLY, IMPLY, CNIMPLY, CIMPLY, XNOR, XOR = 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 #non-trivial boolean operators
def max_one_sat_qubo(clause, varmap, bqm, const):
  i, = clause
  if i > 0: #1-x_i
    add_bqm(bqm, (varmap[i], varmap[i]), -1)
    const += 1
  else: #x_i
    add_bqm(bqm, (varmap[-i], varmap[-i]), 1)
  return bqm, const
#https://arxiv.org/ftp/arxiv/papers/1811/1811.11538.pdf
def max_two_sat_qubo(clause, varmap, bqm, const):
  i, j = clause
  if i > 0 and j > 0: #1-x_i-x_j+x_i*x_j
    add_bqm(bqm, (varmap[i], varmap[j]), 1)
    add_bqm(bqm, (varmap[i], varmap[i]), -1)
    add_bqm(bqm, (varmap[j], varmap[j]), -1)
    const += 1
  elif i < 0 and j < 0: #x_i*x_j
    add_bqm(bqm, (varmap[-i], varmap[-j]), 1)
  elif i < 0: #x_i-x_i*x_j
    add_bqm(bqm, (varmap[-i], varmap[j]), -1)
    add_bqm(bqm, (varmap[-i], varmap[-i]), 1)
  elif j < 0: #x_j-x_i*x_j
    add_bqm(bqm, (varmap[i], varmap[-j]), -1)
    add_bqm(bqm, (varmap[-j], varmap[-j]), 1)
  return bqm, const
def add_bqm(bqm, idx, val):
  if idx[0] > idx[1]: idx = (idx[1], idx[0])
  if not idx in bqm: bqm[idx] = 0
  bqm[idx] += val
  if bqm[idx] == 0: del bqm[idx]
def all_ternary_funcs():
  import operator
  def imply(a, b): return not a or b
  def nimply(a, b): return a and not b
  def cimply(a, b): return a or not b
  def cnimply(a, b): return not a and b
  def nand(a, b): return not (a and b)
  def nor(a, b): return not (a or b)
  UNARY, NOTUNARY = 0, 1
  binoperators = [operator.and_, operator.or_, operator.xor, operator.eq, imply, cimply, nimply, cnimply, nand, nor]
  ternfuncs = [None]*256
  ternfuncs[sum(0<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))] = (0, ALWAYSFALSE)
  ternfuncs[sum(1<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))] = (0, ALWAYSTRUE)
  ternfuncs[sum(a<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))] = (1, UNARY, 0)
  ternfuncs[sum(b<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))] = (1, UNARY, 1)
  ternfuncs[sum(c<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))] = (1, UNARY, 2)
  ternfuncs[sum((1-a)<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))] = (1, NOTUNARY, 0)
  ternfuncs[sum((1-b)<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))] = (1, NOTUNARY, 1)
  ternfuncs[sum((1-c)<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))] = (1, NOTUNARY, 2)
  import itertools
  for i, op1 in enumerate(binoperators):
    val = sum(int(op1(a, b))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
    if ternfuncs[val] is None: ternfuncs[val] = (2, i, (0, 1))
    val = sum(int(op1(b, c))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
    if ternfuncs[val] is None: ternfuncs[val] = (2, i, (1, 2))
    val = sum(int(op1(a, c))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
    if ternfuncs[val] is None: ternfuncs[val] = (2, i, (0, 2))
  def ternary(a, b, c): return b if a else c
  def combmap(a, b, c, val): return (a, b, c) if val==0 else ((a, c, b) if val==1 else ((b, a, c) if val==2 else ((b, c, a) if val==3 else (c, a, b) if val==4 else (c, b, a))))
  def negmap(a, b, c, val): return (a if (val & 4) == 0 else 1-a, b if (val & 2) == 0 else 1-b, c if (val & 1) == 0 else 1-c)
  for comb in range(6):
    for negation in range(8):
      val = sum(ternary(*negmap(*combmap(a, b, c, comb), negation))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
      if ternfuncs[val] is None: ternfuncs[val] = (3, (comb, negation))
  for i, op1 in enumerate(binoperators):
    for j, op2 in enumerate(binoperators):
      val = sum(int(op2(int(op1(a, b)), c))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
      if ternfuncs[val] is None: ternfuncs[val] = (4, (i, j))
      val = sum(int(op1(a, int(op2(b, c))))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
      if ternfuncs[val] is None: ternfuncs[val] = (5, (i, j))
      val = sum(int(op2(int(op1(a, c)), b))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
      if ternfuncs[val] is None: ternfuncs[val] = (6, (i, j))
      val = sum(int(op1(b, int(op2(a, c))))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
      if ternfuncs[val] is None: ternfuncs[val] = (7, (i, j))
  for i, op1 in enumerate(binoperators):
    for j, op2 in enumerate(binoperators):
      for k, op3 in enumerate(binoperators):
        val = sum(int(op3(int(op1(a, b)), int(op2(b, c))))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
        if ternfuncs[val] is None: ternfuncs[val] = (8, (i, j, k))
        val = sum(int(op3(int(op1(a, b)), int(op2(a, c))))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
        if ternfuncs[val] is None: ternfuncs[val] = (9, (i, j, k))
  for i, op1 in enumerate(binoperators):
    for j, op2 in enumerate(binoperators):
      for k, op3 in enumerate(binoperators):
        for l, op4 in enumerate(binoperators):
          val = sum(int(op4(int(op1(a, b)), int(op3(a, int(op2(b, c))))))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
          if ternfuncs[val] is None: ternfuncs[val] = (10, (i, j, k, l))
          val = sum(int(op4(int(op1(a, c)), int(op3(b, int(op2(a, c))))))<<(4*a+2*b+c) for a, b, c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
          if ternfuncs[val] is None: ternfuncs[val] = (11, (i, j, k, l))
  assert sum(1 for x in ternfuncs if x is None) == 0
  sym = [None] * 256
  symnames = ("a", "b", "c")
  binopnames = ("∧", "∨", "⊕ ", "↔", "→", "←", "⇏ ", "⇍ ", "⊼", "⊽")
  binopnamesltx = ("\\wedge ", "\\vee ", "\\oplus ", "\\leftrightarrow ", "\\implies ", "\\leftarrow ", "\\nrightarrow ", "\\nleftarrow ", "\\barwedge ", "\\barvee ")
  negsym = "¬"
  negsymltx = "\\neg "
  alwayssym = ("⊤", "⊥")
  alwayssymltx = ("\\top", "\\bot")
  binopnames, negsym, alwayssym = binopnamesltx, negsymltx, alwayssymltx
  for i, ternfunc in enumerate(ternfuncs):
    if ternfunc[0] == 0:
      sym[i] = (alwayssym[0] if ternfunc[1] == ALWAYSTRUE else alwayssym[1])
    elif ternfunc[0] == 1:
      sym[i] = ("" if ternfunc[1] == UNARY else negsym) + symnames[ternfunc[2]]
    elif ternfunc[0] == 2:
      sym[i] = symnames[ternfunc[2][0]] + binopnames[ternfunc[1]] + symnames[ternfunc[2][1]]
    elif ternfunc[0] == 3:
      comb = combmap(*symnames, ternfunc[1][0])
      sym[i] = ("" if (ternfunc[1][1] & 4) == 0 else negsym) + comb[0] + "?" + ("" if (ternfunc[1][1] & 2) == 0 else negsym) + comb[1] + ":" + ("" if (ternfunc[1][1] & 1) == 0 else negsym) + comb[2]
    elif ternfunc[0] == 4:
      sym[i] = "(" + symnames[0] + binopnames[ternfunc[1][0]] + symnames[1] + ")" + binopnames[ternfunc[1][1]] + symnames[2]
    elif ternfunc[0] == 5:
      sym[i] = symnames[0] + binopnames[ternfunc[1][0]] + "(" + symnames[1] + binopnames[ternfunc[1][1]] + symnames[2] + ")"
    elif ternfunc[0] == 6:
      sym[i] = "(" + symnames[0] + binopnames[ternfunc[1][0]] + symnames[2] + ")" + binopnames[ternfunc[1][1]] + symnames[1]
    elif ternfunc[0] == 7:
      sym[i] = symnames[1] + binopnames[ternfunc[1][0]] + "(" + symnames[0] + binopnames[ternfunc[1][1]] + symnames[2] + ")"
    elif ternfunc[0] == 8:
      sym[i] = "(" + symnames[0] + binopnames[ternfunc[1][0]] + symnames[1] + ")" + binopnames[ternfunc[1][2]] + "(" + symnames[1] + binopnames[ternfunc[1][1]] + symnames[2] + ")"
    elif ternfunc[0] == 9:
      sym[i] = "(" + symnames[0] + binopnames[ternfunc[1][0]] + symnames[1] + ")" + binopnames[ternfunc[1][2]] + "(" + symnames[0] + binopnames[ternfunc[1][1]] + symnames[2] + ")"
    elif ternfunc[0] == 10:
      sym[i] = "(" + symnames[0] + binopnames[ternfunc[1][0]] + symnames[1] + ")" + binopnames[ternfunc[1][3]] + "(" + symnames[0] + binopnames[ternfunc[1][2]] + "(" + symnames[1] + binopnames[ternfunc[1][1]] + symnames[2] + "))"
    elif ternfunc[0] == 11:
      sym[i] = "(" + symnames[0] + binopnames[ternfunc[1][0]] + symnames[2] + ")" + binopnames[ternfunc[1][3]] + "(" + symnames[1] + binopnames[ternfunc[1][2]] + "(" + symnames[0] + binopnames[ternfunc[1][1]] + symnames[2] + "))"
  return sym

def linprog_to_z3(c, Aineq, Bineq, Aeq, Beq, x_bounds, labels=None, updfunc=None, updenum=None, cb=None, constraints=None):
  from z3 import And, Bool, Int, Solver, BoolRef, is_true, CheckSatResult, Z3_L_TRUE, set_param
  set_param('parallel.enable', True)
  s = Solver()
  v = [Bool(labels[i] if not labels is None else None) if x_bounds[i] == (0,1) else Int(name=labels[i] if not labels is None else None) for i in range(len(x_bounds))]
  s.add([And(v[i] >= bound[0], v[i] <= bound[1]) for i, bound in enumerate(x_bounds) if bound != (0,1)])
  if constraints is None:
      ineqconstrparts = [([[v[i], a] for i, a in enumerate(Aineq[j])], Bineq[j]) for j in range(len(Aineq))]
      ineqconstrs = [sum(x*y for x, y in constr if y != 0) <= lim for constr, lim in ineqconstrparts]
      eqconstrparts = [([[v[i], a] for i, a in enumerate(Aeq[j])], Beq[j]) for j in range(len(Aeq))]
      eqconstrs = [sum(x*y for x, y in constr if y != 0) == lim for constr, lim in eqconstrparts]
  else: s.add(constraints(v, None))
  def upd(isIneq, idx, offset, vec):
    cnstr, lim = ineqconstrparts[idx] if isIneq else eqconstrparts[idx]      
    for i, val in enumerate(vec):
      cnstr[offset+i][1] = val
    if isIneq: ineqconstrs[idx] = sum(x*y for x, y in cnstr if y != 0) <= lim
    else: eqconstrs[idx] = sum(x*y for x, y in cnstr if y != 0) == lim
  best = []
  for vals in (None,) if updenum is None else updenum:
    if not updfunc is None: updfunc(*vals, upd)
    if constraints is None: s.reset(); s.add(ineqconstrs + eqconstrs)
    if s.check() == CheckSatResult(Z3_L_TRUE):
      m = s.model()
      res = [int(is_true(m.eval(x))) if isinstance(m.eval(x), BoolRef) else m.eval(x).as_long() for x in v]
    else: res = None
    best.append(res)
  return res if updfunc is None else best
def linprog_to_cplex(c, Aineq, Bineq, Aeq, Beq, x_bounds, labels=None, updfunc=None, updenum=None, cb=None, constraints=None):
  use_cp = False
  #import cplex #with cplex.Cplex() as cpx:
  if use_cp:
    from docplex.cp.model import CpoModel, SOLVE_STATUS_FEASIBLE
  else:
    from docplex.mp.model import Model
  with CpoModel() if use_cp else Model() as m:
    v = [m.binary_var(name=labels[i] if not labels is None else None) if x_bounds[i] == (0,1) else m.integer_var(*x_bounds[i], name=labels[i] if not labels is None else None) for i in range(len(x_bounds))]
    if constraints is None:
      ineqconstrparts = [([[v[i], a] for i, a in enumerate(Aineq[j])], Bineq[j]) for j in range(len(Aineq))]
      ineqconstrs = [sum(x*y for x, y in constr if y != 0) <= lim for constr, lim in ineqconstrparts]
      eqconstrparts = [([[v[i], a] for i, a in enumerate(Aeq[j])], Beq[j]) for j in range(len(Aeq))]
      eqconstrs = [sum(x*y for x, y in constr if y != 0) == lim for constr, lim in eqconstrparts]
      m.add(ineqconstrs)
      m.add(eqconstrs)
    else: (m.add if use_cp else m.add_quadratic_constraints)(constraints(v, None if use_cp else lambda bounds: [m.binary_var(None) if bounds[i] == (0,1) else m.integer_var(*bounds[i], name=None) for i in range(len(bounds))]))
    m.minimize(sum(v[i]*c[i] for i in range(len(c)) if c[i] != 0))
    def upd(isIneq, idx, offset, vec):
      m.remove(ineqconstrs[idx] if isIneq else eqconstrs[idx])
      cnstr, lim = ineqconstrparts[idx] if isIneq else eqconstrparts[idx]      
      for i, val in enumerate(vec):
        cnstr[offset+i][1] = val
      if isIneq: ineqconstrs[idx] = sum(x*y for x, y in cnstr if y != 0) <= lim
      else: eqconstrs[idx] = sum(x*y for x, y in cnstr if y != 0) == lim
      m.add(ineqconstrs[idx] if isIneq else eqconstrs[idx])
    best = []
    for vals in (None,) if updenum is None else updenum:
      if not updfunc is None: updfunc(*vals, upd)
      if not use_cp:
        import multiprocessing
        m.parameters.threads = multiprocessing.cpu_count()
        m.parameters.parallel = -1 #CPX_PARALLEL_OPPORTUNISTIC
        m.parameters.emphasis.mip = 1 #CPX_MIPEMPHASIS_FEASIBILITY
        m.parameters.mip.strategy.probe = 3 #Very aggressive probing level
        m.parameters.mip.strategy.search = 1 #CPX_MIPSEARCH_TRADITIONAL
        m.parameters.mip.strategy.fpheur = 1 #Apply the feasibility pump heuristic with an emphasis on finding a feasible solution
        #m.parameters.optimalitytarget = 2 #CPX_OPTIMALITYTARGET_FIRSTORDER only for continuous
      import os
      res = m.solve(LogPeriod=1000000, SearchType='MultiPoint', SolutionLimit=1, TemporalRelaxation='Off', ConflictRefinerOnVariables='On', DefaultInferenceLevel='Extended', MultiPointNumberOfSearchPoints=len(c), **({} if os.name == 'nt' else {})) if use_cp else m.solve(log_output=True)
      if not use_cp and not res is None and res.is_valid_solution():
        res = [int(round(res.get_value(x))) for x in v]
      elif use_cp and res.get_solve_status() == SOLVE_STATUS_FEASIBLE:
        res = [res.get_var_solution(x).get_value() for x in v]
      else: res = None
      best.append(res)
      if not cb is None: cb(res, *vals)
    return res if updfunc is None else best
def linprog_to_gurobi(c, Aineq, Bineq, Aeq, Beq, x_bounds, labels=None, updfunc=None, updenum=None, cb=None, constraints=None):
  from gurobipy import Env, Model, GRB, read
  with Env(empty=True) as env:
    if not cb is None or False: env.setParam("OutputFlag", 0)
    env.start()
    #with read("curmodel.mps") as m:
    with Model(env=env) as m:
      m.setParam(GRB.Param.Presolve, 2)
      m.setParam(GRB.Param.PreDual, 2)
      m.setParam(GRB.Param.MIPFocus, 1)
      m.setParam(GRB.Param.IntegralityFocus, 1)
      m.setParam(GRB.Param.DegenMoves, 0)
      #m.setParam(GRB.Param.MinRelNodes, 0)
      m.setParam(GRB.Param.Heuristics, 1.0) #maximum feasibility heuristics
      m.setParam(GRB.Param.NoRelHeurWork, 10) #float('inf'))
      #m.setParam(GRB.Param.ImproveStartNodes, 0)
      m.setParam(GRB.Param.ImproveStartTime, 0)
      m.setParam(GRB.Param.ImproveStartGap, float('inf'))
      m.setParam(GRB.Param.ZeroObjNodes, 50000) #2000000000)
      m.setParam(GRB.Param.PumpPasses, 50000) #2000000000)
      m.setParam(GRB.Param.Cuts, 3) #very aggressive
      m.setParam(GRB.Param.RINS, 1)
      if not cb is None: m.setParam(GRB.Param.SolutionLimit, 1)
      v = m.addVars(range(len(c)), lb=[x[0] for x in x_bounds], ub=[x[1] for x in x_bounds], vtype=[GRB.BINARY if x == (0,1) else GRB.INTEGER for x in x_bounds], name=labels)
      m.update()
      for i in range(len(c)):
        if v[i].vtype == GRB.BINARY and not '*' in v[i].varname: v[i].setAttr(GRB.Attr.BranchPriority, 1)
      if constraints is None:
        ineqconstrs = m.addConstrs((sum(v[i]*a for i, a in enumerate(Aineq[j]) if a != 0) <= Bineq[j] for j in range(len(Aineq))), name="ub")
        eqconstrs = m.addConstrs((sum(v[i]*a for i, a in enumerate(Aeq[j]) if a != 0) == Beq[j] for j in range(len(Aeq))), name="eq")
      else: m.addConstrs((x for x in constraints(v, lambda bounds: m.addVars(range(len(bounds)), lb=[x[0] for x in bounds], ub=[x[1] for x in bounds], vtype=[GRB.BINARY if x == (0,1) else GRB.INTEGER for x in bounds]))))
      m.setObjective(sum(v[i]*c[i] for i in range(len(c)) if c[i] != 0), GRB.MINIMIZE)
      def upd(isIneq, idx, offset, vec):
        cnstr = ineqconstrs[idx] if isIneq else eqconstrs[idx]
        for i, val in enumerate(vec):
          m.chgCoeff(cnstr, v[offset+i], val)
      best = []
      for vals in (None,) if updenum is None else updenum:
        if not updfunc is None: updfunc(*vals, upd)
        #m.tune()
        #print(m.getTuneResult)
        #m.update()
        #m.presolve().write("curmodel.mps")
        #m.printStats()
        m.optimize()
        res = [int(round(v.x)) for v in m.getVars()] if m.Status == GRB.OPTIMAL else None
        best.append(res)
        if not cb is None: cb(res, *vals)
      return res if updfunc is None else best
def iqp_ideal_qubo_encoding_allsubs(f, n, nsubs, coeff=None, knownsubs=None):
  import itertools
  d = (n+nsubs)*(n+nsubs+1)//2
  dn = n*(n+1)//2
  C = dn*2 #(1<<(n+nsubs))**2 #d+1+1
  x_bounds = [(-C, C)]*d + [(0, 2*C)] + [(0, 1)]*((1<<n)*nsubs)
  basenum = nary_func_to_num(f, n)
  c = [0]*d + [0] + [1 if (1<<i)&basenum==0 else 0 for i in range(1<<n) for _ in range(nsubs)]
  labels = [f"c_{i}" for i in range(d+1)] + [f"s{j}_{i}" for i in range(1<<n) for j in range(nsubs)]  
  def constraints_from_vars(v, makevars=None):
    constraints = []
    if makevars is None:
      quadvars = [v[d+1+xidx*nsubs+i]*v[d+1+xidx*nsubs+j] for xidx in range(1<<n) for i in range(nsubs) for j in range(i+1, nsubs)]
    else:
      quadvars = makevars([(0, 1)]*((1<<n)*nsubs*(nsubs-1)//2))
      constraints.extend([v[d+1+xidx*nsubs+ij[0]]*v[d+1+xidx*nsubs+ij[1]]==quadvars[xidx*nsubs*(nsubs-1)//2+vidx] for xidx in range(1<<n) for vidx, ij in enumerate((i, j) for i in range(nsubs) for j in range(i+1, nsubs))])
    for xidx, x in enumerate(itertools.product((0,1), repeat=n)):
      verify = 1 - f(*x)
      eq = ([v[i]*z for i, z in enumerate(x) if z] + [v[n+vidx]*x[ij[0]]*x[ij[1]] for vidx, ij in enumerate((i, j) for i in range(n) for j in range(i+1, n)) if x[ij[0]] and x[ij[1]]] +
            [v[dn+i*nsubs+j]*v[d+1+xidx*nsubs+j] for i in range(n) for j in range(nsubs) if x[i]] +
            [v[dn+n*nsubs+j]*v[d+1+xidx*nsubs+j] for j in range(nsubs)] +
            [v[dn+n*nsubs+nsubs+vidx]*quadvars[xidx*nsubs*(nsubs-1)//2+vidx] for vidx, _ in enumerate((i, j) for i in range(nsubs) for j in range(i+1, nsubs))] + [v[d]])
      if verify == 0 or not coeff is None:
        constraints.append(sum(eq) == verify) #==0
      else:
        constraints.append(sum(eq) >= verify) #>=1
      for badsub in itertools.product((0,1), repeat=nsubs):
        if all(y == 0 for y in badsub): continue
        #known: Cst, Cs, Ct so C(1-s)t=Ct-Cst or Cs(1-t)=Cs-Cst or C(1-s)(1-t)=C-Cs-Ct+Cst
        condnegate = lambda z, s: 1-1*z if s else z
        dualnegate = lambda z0, z1, zmix, s0, s1: ((1+zmix-z0-z1 if s0 and s1 else zmix) if s0 == s1 else (z1 if s0 else z0)-zmix)
        eq = ([v[i]*z for i, z in enumerate(x) if z] + [v[n+vidx]*x[ij[0]]*x[ij[1]] for vidx, ij in enumerate((i, j) for i in range(n) for j in range(i+1, n)) if x[ij[0]] and x[ij[1]]] +
              [v[dn+i*nsubs+j]*condnegate(v[d+1+xidx*nsubs+j], badsub[j]) for i in range(n) for j in range(nsubs) if x[i]] +
              [v[dn+n*nsubs+j]*condnegate(v[d+1+xidx*nsubs+j], badsub[j]) for j in range(nsubs)] +
              [v[dn+n*nsubs+nsubs+vidx]*dualnegate(v[d+1+xidx*nsubs+ij[0]], v[d+1+xidx*nsubs+ij[1]], quadvars[xidx*nsubs*(nsubs-1)//2+vidx], badsub[ij[0]], badsub[ij[1]]) for vidx, ij in enumerate((i, j) for i in range(nsubs) for j in range(i+1, nsubs))] + [v[d]])
        constraints.append(sum(eq)*(coeff if not coeff is None else 1) >= (coeff if not coeff is None else 1)) #>=1
    if not knownsubs is None:
      for i, x in enumerate(itertools.product((0,1), repeat=n)):
        for k, sub in enumerate(knownsubs):
          constraints.append(v[d+1+i*nsubs+k]==sub(*x))
    else:
      for k in range(0 if knownsubs is None else len(knownsubs), nsubs):
        for i in range(1<<n):
          if (basenum & (1<<i)) == 0: constraints.append(v[d+1+i*nsubs+k]==0)
        #for l in range(k+1, nsubs):
          #constraints.append(sum((1<<i)*v[d+1+i*nsubs+k] - (1<<i)*v[d+1+i*nsubs+l] for i in range(1<<n)) <= 0)
    return constraints
  cplex, gurobi, z3 = False, True, False
  if cplex or gurobi or z3: 
    if z3: res = linprog_to_z3(c, None, None, None, None, x_bounds, labels, constraints=constraints_from_vars)
    elif cplex: res = linprog_to_cplex(c, None, None, None, None, x_bounds, labels, constraints=constraints_from_vars)
    else: res = linprog_to_gurobi(c, None, None, None, None, x_bounds, labels, constraints=constraints_from_vars)
    return (res[:d+1], [sum(res[d+1+i*nsubs+j]<<i for i in range(1<<n)) for j in range(nsubs)]) if not res is None else res
def ilp_ideal_qubo_encoding_allsubs(f, n, nsubs, coeff=None, knownsubs=None):
  from scipy.optimize import linprog
  import numpy as np
  import itertools
  d = (n+nsubs)*(n+nsubs+1)//2
  dn = n*(n+1)//2
  C = dn*2 #(1<<(n+nsubs))**2 #d+1+1
  x_bounds = [(-C, C)]*d + [(0, 2*C)] + [(0, 1)]*((1<<n)*nsubs) + [(0, 1)]*((1<<n)*nsubs*(nsubs-1)//2) + [(-C, C)]*((1<<n)//2*nsubs*(2+n)) + [(-C, C)]*((1<<n)*nsubs*(nsubs-1)//2) + [(-C, C)]*((1<<n)*nsubs*(nsubs-1))
  assert d == dn + nsubs*n + nsubs + nsubs*(nsubs-1)//2 #dimensionality check
  #remaining dimensions: [1<<n X nsubs] [1<<n X nsubs*(nsubs-1)//2] [(1<<n)//2*n X nsubs] [1<<n X nsubs] [1<<n X nsubs*(nsubs-1)//2] [1<<n X nsubs*(nsubs-1)//2 X 2]
  basenum = nary_func_to_num(f, n)
  c = [0]*d + [0] + [1 if (1<<i)&basenum==0 else 0 for i in range(1<<n) for _ in range(nsubs)] + [0]*((1<<n)*nsubs*(nsubs-1)//2) + [0]*((1<<n)//2*nsubs*(2+n)) + [0]*((1<<n)*nsubs*(nsubs-1)//2) + [0]*((1<<n)*nsubs*(nsubs-1))
  #c = [0]*d + [0] + [1]*((1<<n)*nsubs) + [0]*((1<<n)*nsubs*(nsubs-1)//2) + [0]*((1<<n)//2*nsubs*(2+n)) + [0]*((1<<n)*nsubs*(nsubs-1)//2) + [0]*((1<<n)*nsubs*(nsubs-1))
  labels = [f"c_{i}" for i in range(d+1)] + [f"s{j}_{i}" for i in range(1<<n) for j in range(nsubs)] + [f"s_{j}_{i}*s_{k}_{i}" for i in range(1<<n) for j in range(nsubs) for k in range(j+1, nsubs)]
  labels += [f"M_{j}_{i}" for i in range((1<<n)//2*n) for j in range(nsubs)] + [f"c_{dn+n*nsubs+j}*s_{j}_{i}" for i in range(1<<n) for j in range(nsubs)]
  labels += [f"c_{dn+(n+1)*nsubs}_{j},{k}*s_{j}_{i}*s_{k}_{i}" for i in range(1<<n) for j in range(nsubs) for k in range(j+1, nsubs)]
  labels += [f"c_{dn+(n+1)*nsubs}_{j},{k}*s_{j if l==0 else k}_{i}" for i in range(1<<n) for j in range(nsubs) for k in range(j+1, nsubs) for l in range(2)]
  assert len(labels) == len(c)
  #print(labels)
  Aeq, Beq, Aineq, Bineq = [], [], [], []
  nonzeroidx = 0
  #Fortet inequalities (binary specification of McCormick) linearize product of two binary variables z=xy  z<=x z<=y z>=x+y-1 z-x<=0 z-y<=0 x+y-z<=1 (0<=z<=1 bound)
  def binary_quadratic(xidx, yidx, zidx): #-z-x<=0 -z+x+y<=1 z-x<=0 z+x-y<=1
    assert xidx < (1<<n)*nsubs
    assert yidx < (1<<n)*nsubs
    assert zidx < (1<<n)*nsubs*(nsubs-1)//2+(1<<n)//2*nsubs*(2+n)+(1<<n)*nsubs*(nsubs-1)//2+(1<<n)*nsubs*(nsubs-1)
    x = [0]*(d+1) + [1 if idx == xidx else 0 for idx in range((1<<n)*nsubs)]
    y = [0]*(d+1) + [1 if idx == yidx else 0 for idx in range((1<<n)*nsubs)]
    z = [1 if idx == zidx else 0 for idx in range((1<<n)*nsubs*(nsubs-1)//2+(1<<n)//2*nsubs*(2+n)+(1<<n)*nsubs*(nsubs-1)//2+(1<<n)*nsubs*(nsubs-1))]
    Aineq.append([-q for q in x] + z); Bineq.append(0)
    Aineq.append([-q for q in y] + z); Bineq.append(0)
    Aineq.append([q+r for q,r in zip(x,y)] + [-q for q in z]); Bineq.append(1)
  #McCormick inequalities (relaxation if not integers): for z=Ms  -z-Cs<=0 -z+Cs+M<=C z-Cs<=0 z+Cs-M<=C  
  #if s==0 then z>=0 z>=M-C z<=0 z<=C+M, if s==1 then z>=-C z>=M z<=C z<=M
  def mccormick_relaxation(Midx, sidx, zidx):
    assert Midx < d-dn
    assert sidx < ((1<<n)*nsubs+(1<<n)*nsubs*(nsubs-1)//2)
    assert zidx < (1<<n)//2*nsubs*(2+n)+(1<<n)*nsubs*(nsubs-1)//2+(1<<n)*nsubs*(nsubs-1)
    M = [0]*dn + [1 if idx == Midx else 0 for idx in range(d-dn)] + [0]
    s = [C if idx == sidx else 0 for idx in range((1<<n)*nsubs+(1<<n)*nsubs*(nsubs-1)//2)]
    z = [1 if idx == zidx else 0 for idx in range((1<<n)//2*nsubs*(2+n)+(1<<n)*nsubs*(nsubs-1)//2+(1<<n)*nsubs*(nsubs-1))]
    Aineq.append([0]*(d+1) + [-q for q in s] + [-q for q in z]); Bineq.append(0)
    Aineq.append(M + s + [-q for q in z]); Bineq.append(C)
    Aineq.append([0]*(d+1) + [-q for q in s] + z); Bineq.append(0)
    Aineq.append([-q for q in M] + s + z); Bineq.append(C)
  for xidx, x in enumerate(itertools.product((0,1), repeat=n)):
    verify = 1 - f(*x)
    nzmax = nonzeroidx + sum(x)
    eq = list(x) + [x[i]*x[j] for i in range(n) for j in range(i+1, n)] + [0]*(d-dn) + [1] + [0]*((1<<n)*nsubs+(1<<n)*nsubs*(nsubs-1)//2) + [1 if i >= nonzeroidx and i < nzmax else 0 for i in range((1<<n)//2*n) for j in range(nsubs)] + [1 if xidx == i else 0 for i in range(1<<n) for j in range(nsubs)] + [1 if xidx == k else 0 for k in range(1<<n) for i in range(nsubs) for j in range(i+1, nsubs)] + [0]*((1<<n)*nsubs*(nsubs-1))
    #eq = [x_i, x_j, x_k, x_i*x_j, x_i*x_k, x_j*x_k, x_i*s, x_j*s, x_k*s, s, 1]
    if verify == 0 or not coeff is None:
      Aeq.append(eq); Beq.append(verify) #==0
    else:
      Aineq.append([-y for y in eq]); Bineq.append(-verify) #>=1
    for badsub in itertools.product((0,1), repeat=nsubs):
      if all(y == 0 for y in badsub): continue
      #known: Cst, Cs, Ct so C(1-s)t=Ct-Cst or Cs(1-t)=Cs-Cst or C(1-s)(1-t)=C-Cs-Ct+Cst
      eq = list(x) + [x[i]*x[j] for i in range(n) for j in range(i+1, n)] + [1 if badsub[i] and x[j] else 0 for i in range(nsubs) for j in range(n)] + [1 if badsub[i] else 0 for i in range(nsubs)] + [1 if badsub[i] and badsub[j] else 0 for i in range(nsubs) for j in range(i+1, nsubs)] + [1] + [0]*((1<<n)*nsubs+(1<<n)*nsubs*(nsubs-1)//2) + [(-1 if badsub[j] else 1) if i >= nonzeroidx and i < nzmax else 0 for i in range((1<<n)//2*n) for j in range(nsubs)] + [(-1 if badsub[j] else 1) if xidx == i else 0 for i in range(1<<n) for j in range(nsubs)] + [(-1 if badsub[i] != badsub[j] else 1) if xidx == k else 0 for k in range(1<<n) for i in range(nsubs) for j in range(i+1, nsubs)] + [(1 if badsub[i] != badsub[j] else -1) if xidx == k and (badsub[i] and badsub[j] or badsub[i] and l==1 or badsub[j] and l==0) else 0 for k in range(1<<n) for i in range(nsubs) for j in range(i+1, nsubs) for l in range(2)]
      if coeff == 1:
        Aeq.append(eq); Beq.append(verify)
      else:
        Aineq.append([-y*(coeff if not coeff is None else 1) for y in eq]); Bineq.append(-verify*(coeff if not coeff is None else 1)) #>=1
    for i, xi in enumerate(x):
      if xi == 0: continue
      for j in range(nsubs):
        mccormick_relaxation(n*j+i, xidx*nsubs+j, nonzeroidx*nsubs+j)
      nonzeroidx += 1
    offset = 0
    for i in range(nsubs):
      mccormick_relaxation(n*nsubs+i, xidx*nsubs+i, (1<<n)//2*nsubs*n + xidx*nsubs+i)
      for j in range(i+1, nsubs):
        zidx = xidx*nsubs*(nsubs-1)//2+offset
        binary_quadratic(xidx*nsubs+i, xidx*nsubs+j, zidx)
        mccormick_relaxation((n+1)*nsubs+offset, (1<<n)*nsubs+zidx, (1<<n)//2*nsubs*(2+n)+xidx*nsubs*(nsubs-1)//2+offset)
        mccormick_relaxation((n+1)*nsubs+offset, xidx*nsubs+i, (1<<n)//2*nsubs*(2+n) + (1<<n)*nsubs*(nsubs-1)//2 + xidx*nsubs*(nsubs-1)//2*2 + offset*2)
        mccormick_relaxation((n+1)*nsubs+offset, xidx*nsubs+j, (1<<n)//2*nsubs*(2+n) + (1<<n)*nsubs*(nsubs-1)//2 + xidx*nsubs*(nsubs-1)//2*2 + offset*2+1)
        offset += 1
  if not knownsubs is None:
    for xidx, x in enumerate(itertools.product((0,1), repeat=n)):
      for idx, sub in enumerate(knownsubs):
        Aeq.append([0]*d + [0] + [1 if i==xidx and j==idx else 0 for i in range(1<<n) for j in range(nsubs)] + [0]*((1<<n)*nsubs*(nsubs-1)//2) + [0]*((1<<n)//2*nsubs*(2+n)) + [0]*((1<<n)*nsubs*(nsubs-1)//2) + [0]*((1<<n)*nsubs*(nsubs-1)))
        Beq.append(sub(*x))
  else:
    for k in range(nsubs):
      for l in range(k+1, nsubs):
        Aineq.append([0]*d + [0] + [(1<<i if j==k else -1<<i) if j==k or j==l else 0 for i in range(1<<n) for j in range(nsubs)] + [0]*((1<<n)*nsubs*(nsubs-1)//2) + [0]*((1<<n)//2*nsubs*(2+n)) + [0]*((1<<n)*nsubs*(nsubs-1)//2) + [0]*((1<<n)*nsubs*(nsubs-1))); Bineq.append(0)
  #for row, brow in zip(Aineq, Bineq): print('+'.join(f"{c}*{l}" for l, c in zip(labels, row) if c != 0) + "<=" + str(brow))
  #for row, brow in zip(Aeq, Beq): print('+'.join(f"{c}*{l}" for l, c in zip(labels, row) if c != 0) + "==" + str(brow))
  #print("Constants:", d+1, "Substitution variables:", (1<<n)*nsubs, "Total:", len(c))
  if False:
    subs = [lambda a, b, c, d: a | b, lambda a, b, c, d: c | d]
    compare = ilp_ideal_qubo_encoding(f, n, [subs])[0].tolist()
    compare += [subs[j](*x) for x in itertools.product((0,1), repeat=n) for j in range(nsubs)]
    compare += [subs[j](*x)*subs[k](*x) for x in itertools.product((0,1), repeat=n) for j in range(nsubs) for k in range(j+1, nsubs)]
    nonzeroidx = 0
    for x in itertools.product((0,1), repeat=n):
      for i, xi in enumerate(x):
        if xi == 0: continue
        for j in range(nsubs):
          compare.append(subs[j](*x)*compare[dn+n*j+i])
      nonzeroidx += 1
    compare += [subs[j](*x)*compare[dn+n*nsubs+j] for i in itertools.product((0,1), repeat=n) for j in range(nsubs)]
    compare += [subs[j](*x)*subs[k](*x)*compare[dn+(n+1)*nsubs+0] for x in itertools.product((0,1), repeat=n) for j in range(nsubs) for k in range(j+1, nsubs)]  
    compare += [subs[j if l == 0 else k](*x)*compare[dn+(n+1)*nsubs+0] for x in itertools.product((0,1), repeat=n) for j in range(nsubs) for k in range(j+1, nsubs) for l in range(2)]  
    for row, brow in zip(Aeq, Beq): print(sum(val*l for val, l in zip(compare, row)) == brow, sum(val*l for val, l in zip(compare, row)), brow)
    for row, brow in zip(Aineq, Bineq): print(sum(val*l for val, l in zip(compare, row)) <= brow, sum(val*l for val, l in zip(compare, row)), brow)
    print([sum(compare[d+1+i*nsubs+j]<<i for i in range(1<<n)) for j in range(nsubs)])
    print(compare[:d+1], ilp_ideal_qubo_encoding(f, n, [[lambda a, b, c, d: get_quaternary_func(12850)(a, b, c, d), lambda a, b, c, d: get_quaternary_func(28785)(a, b, c, d)]]))
    print(ilp_ideal_qubo_encoding(f, n, [[lambda a, b, c, d: get_quaternary_func(12850)(a, b, c, d), lambda a, b, c, d: get_quaternary_func(28785)(a,b,c,d)]], None))
  if n <= 5 or nsubs <= 1:
    cplex, gurobi, z3, glpk = False, False, False, False
  else:
    cplex, gurobi, z3, glpk = True, True, False, False
  if glpk:
    from glpk import glpk
    res = glpk(c, A_ub=Aineq, b_ub=Bineq, A_eq=None if len(Aeq) == 0 else Aeq, b_eq=None if len(Beq) == 0 else Beq, bounds=x_bounds, scale=False, solver="mip",
               mip_options={'nomip': False, 'backtrack':'bestp', 'branch':'pcost', 'cuts':'all', 'fpump':True, 'intcon': [i for i, x in enumerate(x_bounds) if x!=(0,1)], 'bincon': [i for i, x in enumerate(x_bounds) if x==(0,1)]})
    print(res.x)
  elif cplex or gurobi or z3: 
    if z3: res = linprog_to_z3(c, Aineq, Bineq, Aeq, Beq, x_bounds, labels)
    elif cplex: res = linprog_to_cplex(c, Aineq, Bineq, Aeq, Beq, x_bounds, labels)
    else: res = linprog_to_gurobi(c, Aineq, Bineq, Aeq, Beq, x_bounds, labels)
    return (res[:d+1], [sum(res[d+1+i*nsubs+j]<<i for i in range(1<<n)) for j in range(nsubs)]) if not res is None else res
  else:
    res = linprog(c, A_ub=Aineq, b_ub=Bineq, A_eq=None if len(Aeq) == 0 else Aeq, b_eq=None if len(Beq) == 0 else Beq, bounds=x_bounds, method="highs", integrality=1)
    if not res.x is None:
      #for row, brow in zip(Aeq, Beq): print(sum(val*l for val, l in zip(np.rint(res.x).astype(np.int64), row)) == brow, sum(val*l for val, l in zip(np.rint(res.x).astype(np.int64), row)), brow)
      #for row, brow in zip(Aineq, Bineq): print(sum(val*l for val, l in zip(np.rint(res.x).astype(np.int64), row)) <= brow, sum(val*l for val, l in zip(np.rint(res.x).astype(np.int64), row)), brow)
      print(np.rint(res.x).astype(np.int64))
    #print(res)
    return np.rint(res.x[:d+1]).astype(np.int64) if not res.x is None else None, [sum(np.rint(res.x[d+1+i*nsubs+j]).astype(np.int64)<<i for i in range(1<<n)) for j in range(nsubs)] if not res.x is None else res.x
def ilp_ideal_qubo_encoding(f, n, subs, coeff=None, nomin=False, funcevals=None): #coeff==1 is not relaxed
  from scipy.optimize import linprog
  import numpy as np
  import itertools
  nsubs = len(subs[0])
  d = (n+nsubs)*(n+nsubs+1)//2
  dn = n*(n+1)//2
  C = dn*2 #dn*2 #(1<<(n+nsubs))**2 #d+1+1
  x_bounds = [(-C, C)]*d + [(0, 2*C)] if nomin else [(-C, C)]*d+[(0, 2*C)] + [(0, C)]*d + [(0, 1)]*d + [(0, 1)]*nsubs #[(-n+1, n-1)]*n+[(0,n//2)]*(dn-n)+[(-C, C)]*(d-dn+1)
  allbest = []
  #zero or one zero range where C=abs(max(x_bounds)): (xn, xCn) while two zeros (x(n-1), xC(n-1)) so xC(n-1) < xn in this case x*12*9<x*10
  c = [0]*d + [0] + ([] if nomin else [1]*d + [d*C]*d + [d*d*C]*nsubs) #introduce new variables to do absolute value abs(x_i)=y_i where -y_i<=x_i<=y_i
  #sign(abs(x_i))=sign(y_i)=z_i where y_i<=Cz_i
  Aeq, Beq, Aineq, Bineq = [], [], [], []
  fdict = {}
  for x in itertools.product((0,1), repeat=n) if funcevals is None else funcevals:
    verify = 1 - f(*x)    
    fdict[x] = (verify, len(Aeq), len(Aineq))
    eq = list(x) + [x[i]*x[j] for i in range(n) for j in range(i+1, n)] + [0]*(d-dn) + [1] + ([] if nomin else [0]*d + [0]*d + [0]*nsubs)
    #eq = [x_i, x_j, x_k, x_i*x_j, x_i*x_k, x_j*x_k, x_i*s, x_j*s, x_k*s, s, 1]
    if verify == 0 or not coeff is None:
      Aeq.append(eq); Beq.append(verify) #==0
    else:
      Aineq.append([-y for y in eq]); Bineq.append(-verify) #>=1
    for badsub in itertools.product((0,1), repeat=nsubs):
      if all(y == 0 for y in badsub): continue
      eq = list(x) + [x[i]*x[j] for i in range(n) for j in range(i+1, n)] + [0]*(d-dn) + [1] + ([] if nomin else [0]*d + [0]*d + [0]*nsubs)
      if coeff == 1:
        Aeq.append(eq); Beq.append(verify)
      else:
        Aineq.append([-y*(coeff if not coeff is None else 1) for y in eq]); Bineq.append(-verify*(coeff if not coeff is None else 1)) #>=1
  if not nomin:
    for idx in range(d):
      neg = [-1 if i == idx else 0 for i in range(d)]
      pos = [1 if i == idx else 0 for i in range(d)]
      Aineq.append(neg + [0] + neg + [0]*d + [0]*nsubs); Bineq.append(0) #-x_i-y_i<=0
      Aineq.append(pos + [0] + neg + [0]*d + [0]*nsubs); Bineq.append(0) #x_i-y_i<=0
      Aineq.append([0]*d + [0] + pos + [C*x for x in neg] + [0]*nsubs); Bineq.append(0) #y_i-Cz_i<=0
    for idx in range(nsubs):
      pos = [0]*dn + [y for i in range(nsubs) for y in [1 if i == idx else 0 for _ in range(n)]] + [1 if i == idx else 0 for i in range(nsubs)] + [1 if i == idx or j == idx else 0 for i in range(nsubs) for j in range(i+1, nsubs)]
      Aineq.append([0]*d + [0] + [0]*d + pos + [-(nsubs+n) if i == idx else 0 for i in range(nsubs)]); Bineq.append(0) #y_i-Cz_i<=0
  def updfunc(idx, sub, upd=None):
    for x in itertools.product((0,1), repeat=n) if funcevals is None else funcevals:
      verify, eqidx, ineqidx = fdict[x]
      s = [sf(*x) for sf in sub]
      eq = [y for si in s for y in [z*si for z in x]] + s + [s[i]*s[j] for i in range(nsubs) for j in range(i+1, nsubs)]
      if verify == 0 or not coeff is None:
        if not upd is None: upd(0, eqidx, dn, eq)
        else: Aeq[eqidx][dn:d] = eq
      else:
        if not upd is None: upd(1, ineqidx, dn, [-y for y in eq])
        else: Aineq[ineqidx][dn:d] = [-y for y in eq]
        ineqidx+=1
      for badsub in itertools.product((0,1), repeat=nsubs):
        if all(y == 0 for y in badsub): continue
        bs = [1 - si if bs else si for si, bs in zip(s, badsub)] #bad substitution
        eq = [y for si in bs for y in [z*si for z in x]] + bs + [bs[i]*bs[j] for i in range(nsubs) for j in range(i+1, nsubs)]
        if not upd is None: upd(1, ineqidx, dn, [-y*(coeff if not coeff is None else 1) for y in eq])
        else: Aineq[ineqidx][dn:d] = [-y*(coeff if not coeff is None else 1) for y in eq]
        ineqidx+=1
  def pfunc(res, idx, sub):
    if len(subs) > 256 and not res is None: print(res, [(nary_func_to_num(sf, n), n) for sf in sub]) #[nary_func_to_num(sf, n) for sf in sub], [num_func_to_formula(nary_func_to_num(sf, n), n) for sf in sub]
    #if n > 3 and not res is None: assert valid_nary_combs(n, [set(nary_func_to_count_nums(sf, n)) for sf in sub]), [nary_func_to_count_nums(sf, n) for sf in sub]
  cplex, gurobi = False, False
  if cplex or gurobi:
    if cplex: res = linprog_to_cplex(c, Aineq, Bineq, Aeq, Beq, x_bounds, updfunc=updfunc, updenum=enumerate(subs), cb=pfunc)
    else: res = linprog_to_gurobi(c, Aineq, Bineq, Aeq, Beq, x_bounds, updfunc=updfunc, updenum=enumerate(subs), cb=pfunc)
    allbest.extend([x[:d+1] if not x is None else x for x in res])
  else:
    for idx, sub in enumerate(subs):
      updfunc(idx, sub)
      print(n, nsubs, len(c), len(Aeq), len(Aineq)-d*3)
      res = linprog(c, A_ub=None if len(Aineq) == 0 else Aineq, b_ub=None if len(Bineq) == 0 else Bineq, A_eq=None if len(Aeq) == 0 else Aeq, b_eq=None if len(Beq) == 0 else Beq, bounds=x_bounds, method="highs", integrality=1)
      if False:
        print("\setcounter{MaxMatrixCols}{", len(c)-(d-1)*2+2, "}", "\\begin{align*}",
              "f(a,b,c)=a\\oplus b\\oplus c,", "s(a,b,c)=a\\vee b,",
              "n=", n, ", ", "n_s=", nsubs, ", ", "|c|=", len(c), ", ", "|A_{eq}|=|B_{eq}|=", len(Aeq), ", ", "|A_{ub}|=|B_{ub}|=", len(Aineq), ",\\\\",
              "\\min", "\\begin{bmatrix}", " & ".join(str(x) for x in c[:d+2] + ["...", c[d+1+d], "..."] + c[-nsubs:]), "\\end{bmatrix}^\\intercal X,\\\\",            
              "\\begin{bmatrix}", " & ".join(str(x if  x=="..." else x[0]) for x in x_bounds[:d+2] + ["...", x_bounds[d+1+d], "..."] + x_bounds[-nsubs:]), "\\end{bmatrix}", "\\le X \\le\\\\",
              "\\begin{bmatrix}", " & ".join(str(x if  x=="..." else x[1]) for x in x_bounds[:d+2] + ["...", x_bounds[d+1+d], "..."] + x_bounds[-nsubs:]), "\\end{bmatrix},\\\\",
              "\\begin{bmatrix}", "\\\\ ".join([" & ".join(str(z) for z in x[:d+2] + ["...", x[d+1+d], "..."] + x[-nsubs:]) for x in Aeq]), "\\end{bmatrix}", "X=", "\\begin{bmatrix}", "\\\\ ".join(str(x) for x in Beq), "\\end{bmatrix},\\\\",
              "\\begin{bmatrix}", "\\\\ ".join([" & ".join(str(z) for z in x[:d+2] + ["...", x[d+1+d], "..."] + x[-nsubs:]) for x in Aineq[:-(d-1)*3-nsubs] + [["..."]*len(c)] + Aineq[-nsubs:]]), "\\end{bmatrix}", "X=", "\\begin{bmatrix}", "\\\\ ".join(str(x) for x in Bineq[:-(d-1)*3-nsubs] + ["..."] + Bineq[-nsubs:]), "\\end{bmatrix},\\\\",
              "X=", "\\begin{bmatrix}", " & ".join(str(x if  x=="..." else np.rint(x).astype(np.int64)) for x in (*res.x[:d+2], "...", res.x[d+1+d], "...", *res.x[-nsubs:])), "\\end{bmatrix}",
              "\\end{align*}")
      allbest.append(np.rint(res.x[:d+1]).astype(np.int64) if not res.x is None else res.x) #res.success
      if not allbest[-1] is None and (allbest[-1][:d] == 0).all(): allbest[-1][d] = np.sign(allbest[-1][d])
      pfunc(allbest[-1], idx, sub)
  return allbest
#print(ilp_ideal_qubo_encoding(lambda a, b, c: a ^ b ^ c, 3, subs=[[lambda a, b, c: a | b]])); assert False
def bitcount(x):
  b = 0
  while x > 0: x&=x-1; b+=1
  return b
def find_ideal_qubo_encoding_allsubs_tree(f, n, nsubs, coeff=None, knownsubs=[]):
  import itertools, functools, operator
  extensions = list(sorted(itertools.product((1,0), repeat=nsubs), key=lambda x: all(z==x[0] for z in x[1:])))
  #extensions = list(itertools.product((0,1), repeat=nsubs))
  s = [[x] for x in extensions]
  basenum = nary_func_to_num(f, n)
  #bitorder = list(reversed([i for j in range(n+1) for i in range(1<<n) if j==sum(1 if i&(1<<k)!=0 else 0 for k in range(n))]))
  bitorder = list(([i for l in range(2) for j in range(n+1) for i in range(1<<n) if (((basenum&(1<<i))!=0)^(l==0)) and (j==sum(1 if i&(1<<k)!=0 else 0 for k in range(n)))]))
  #bitorder = list([i for j in range(2) for i in range((1<<n)) if (basenum&(1<<i)!=0) ^ (j==0)])
  #bitorder = list([i for j in range(2) for i in range((1<<n)-1,-1,-1) if (basenum&(1<<i)!=0) ^ (j==0)])
  #bitorder = list(range(1<<n))
  funcevals = list(itertools.product((0,1), repeat=n))
  funcevals = [funcevals[i] for i in bitorder]
  print(hex(basenum), bitorder, funcevals)
  while len(s) != 0:
    subs = s.pop()
    tots = [sum(x[i]<<bitorder[j] for j, x in enumerate(subs)) for i in range(nsubs)]
    if any(tots[i] > tots[i+1] for i in range(nsubs-1)): continue
    if any((x & ~basenum) != 0 for x in tots): continue
    if bitcount(functools.reduce(operator.and_, (basenum, *tots))) >= bitcount(basenum)*2//3: continue
    #print(len(subs), hex(basenum), [hex(x) for x in tots])
    check = ilp_ideal_qubo_encoding(f, n, [knownsubs+[get_nary_func(x, n) for x in tots]], coeff, True, funcevals=funcevals[:len(subs)])[0]
    if not check is None:
      if len(subs) == 1<<n: print("Solution:", [hex(x) for x in tots], subs, check, hex(basenum), bitcount(basenum), [bitcount(x) for x in tots], [bitcount(x&basenum) for x in tots], [bitcount(tots[i]&tots[j]) for i in range(nsubs) for j in range(i+1, nsubs)])
      else: #eliminate symmetric cases in 2 variables by making the first always larger than the second
        s.extend([subs + [x] for x in extensions])
def find_ideal_qubo_encoding(f, subs):
  import itertools
  allbest = []
  for idx, sub in enumerate(subs):
    best, num_zeros = [], -1
    for C in itertools.product((0,-1,1,-2,2,3), repeat=10):
      z = sum(1 for x in C[:-1] if x == 0)
      if z < num_zeros: continue
      c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = C; c_10, firstcheck = -4, True
      for x_i, x_j, x_k in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)):
        verify = 1 - f(x_i, x_j, x_k)
        s = sub(x_i, x_j, x_k)
        check = c_0*x_i+c_1*x_j+c_2*x_k+c_3*x_i*x_j+c_4*x_i*x_k+c_5*x_j*x_k+c_6*x_i*s+c_7*x_j*s+c_8*x_k*s+c_9*s+c_10
        if firstcheck:
          if verify == 0 and check < 0: c_10 -= check; check = 0
          elif verify != 0 and check <= 0: c_10 -= check - 1; check = 1
          if verify == 0: firstcheck = False
        #if verify != check: break
        if (verify == 0) != (check == 0) or check < verify: break
        s = 1 - s #bad substitution
        check = c_0*x_i+c_1*x_j+c_2*x_k+c_3*x_i*x_j+c_4*x_i*x_k+c_5*x_j*x_k+c_6*x_i*s+c_7*x_j*s+c_8*x_k*s+c_9*s+c_10
        if check < verify: break
      else:
        if z > num_zeros or z == num_zeros and sum(abs(x) for x in C) < sum(abs(x) for x in best[-1][:-1]): best, num_zeros = [C + (c_10,)], z
        elif z == num_zeros and sum(abs(x) for x in C) == sum(abs(x) for x in best[-1][:-1]):
          best.append(C + (c_10,))
    allbest.append((idx, best))
  return allbest
allbinarysubs = [lambda a, b, c: a & b, lambda a, b, c: 1-(a & b), lambda a, b, c: a | b, lambda a, b, c: 1-(a | b),
                                lambda a, b, c: a ^ b, lambda a, b, c: 1-(a ^ b), lambda a, b, c: a & (1-b), lambda a, b, c: (1-a) | b,
                                lambda a, b, c: (1-a) & b, lambda a, b, c: a | (1-b)]
allternarysat = [lambda a, b, c: a | b | c, lambda a, b, c: a | b | (1-c), lambda a, b, c: a | (1-b) | c, lambda a, b, c: (1-a) | b | c,
                 lambda a, b, c: a | (1-b) | (1-c), lambda a, b, c: (1-a) | b | (1-c), lambda a, b, c: (1-a) | (1-b) | c, lambda a, b, c: (1-a) | (1-b) | (1-c)]
def all_cnf_binary_ops():
  from sympy import Symbol
  from sympy.logic.boolalg import to_cnf, Xor, Nand, Nor, Xnor, Implies
  a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
  print(to_cnf(a & b), to_cnf(Nand(a, b)), to_cnf(a | b), to_cnf(Nor(a, b)), to_cnf(Xor(a, b)), to_cnf(Xnor(a, b)), to_cnf(Implies(a, b)), to_cnf(~Implies(a, b)), to_cnf(Implies(b, a)), to_cnf(~Implies(b, a)))
def get_ternary_func(val): return lambda a, b, c: 1 if (val & (1<<(a * 4 + b * 2 + c))) != 0 else 0
def get_quaternary_func(val): return lambda a, b, c, d: 1 if (val & (1<<(a * 8 + b * 4 + c * 2 + d))) != 0 else 0
def get_quinary_func(val): return lambda a, b, c, d, e: 1 if (val & (1<<(a * 16 + b * 8 + c * 4 + d * 2 + e))) != 0 else 0
def get_nary_func(val, n):
  return lambda *args: 1 if val & (1<<(sum(args[n-1-i]<<i for i in range(n)))) != 0 else 0
def get_nary_count_func(counts):
  return lambda *args: 1 if sum(args) in counts else 0
def nary_func_to_num(f, n):
  import itertools
  return sum(f(*x)<<i for i, x in enumerate(itertools.product((0,1), repeat=n)))
#assert nary_func_to_num(lambda a, b, c: c ^ (a & (1-b)), 3) == nary_func_to_num(get_nary_func(nary_func_to_num(lambda a, b, c: c ^ (a & (1-b)), 3), 3), 3)
def nary_func_to_count_nums(f, n):
  import itertools
  counts = {(1 if f(*x) else -1)*(sum(x)+1) for x in itertools.product((0,1), repeat=n)}
  assert len(counts & {-x for x in counts}) == 0
  return {x-1 for x in counts if x>0}
def all_nary_count_combs(n):
  import itertools
  for i in range(1, n+1):
    for comb in itertools.combinations(range(1, n+1), i):
      #assert nary_func_to_count_nums(get_nary_count_func(set(comb)), n) == set(comb)
      yield set(comb)
def subsets_k(collection, k): yield from partition_k(collection, k, k)
def partition_k(collection, min, k, maxsize=2):  
  if len(collection) == 1:
    yield [collection]
    return
  first = collection[0]
  for smaller in partition_k(collection[1:], min - 1, k):
    if any(len(x)>maxsize for x in smaller): continue
    if len(smaller) > k: continue
    if len(smaller) >= min:
      for n, subset in enumerate(smaller):
        if len(subset) == maxsize: continue
        yield smaller[:n] + [[first] + subset] + smaller[n+1:]
    if len(smaller) < k: yield [[first]] + smaller
def partition_groups(collection, groups):
  if len(groups) == 1: yield [collection]; return
  import itertools
  for selection in itertools.combinations(range(len(collection)), groups[0]):
    yield from [[[collection[i] for i in selection]] + x for x in partition_groups([collection[i] for i in range(len(collection)) if not i in selection], groups[1:])]
def valid_nary_combs(n, combs):
  import functools, operator, itertools
  negset = frozenset({0}) #all case and the singular negative case (for 3CNF it is 0)
  combs = [set(comb) - negset for comb in combs]
  basesets = {frozenset(set(range(n+1)) - set(functools.reduce(operator.or_, combs)) - negset), negset}
  for i in range(len(combs), 1, -1):
    for subcombs in itertools.combinations(combs, i):
      inallsets = frozenset(functools.reduce(operator.and_, subcombs))
      if len(inallsets) != 0:
        basesets.add(inallsets)
        combs = [comb - inallsets for comb in combs]
  #if sum(len(x)==0 for x in combs) > len(combs)-2: return False
  basesets |= {frozenset(comb) for comb in combs}
  return all(len(x) <= 2 for x in basesets)
def stirling_numbers_second_kind(n, k):
  import math
  return sum((-1 if ((k-i) & 1) != 0 else 1)*(i**n)*math.comb(k, i) for i in range(k+1))//math.factorial(k)
#C0*z+C1*z*(z-1)//2+C2*z(z in sums1)+C3*z*(z in sums2)+C4*(z in sums1)+C5*(z in sums2)+C6(z in sums2)(z in sums1)+C7
#z=sum(x), C0=-n+1, C1=1, C7=n(n-1)//2, C6=C3*C4
def cnf_linalg_solve(n):
  nsubs = (n-1).bit_length()-1
  b = [-n+1, 1, n*(n-1)//2]
  subvars = [1<<i for i in range(nsubs-1, -1, -1)]
  idx, cumsum = len(subvars)-1, n-1-(1<<nsubs)
  while cumsum != 0:
    if cumsum > subvars[idx]: cumsum -= subvars[idx]; subvars[idx] <<= 1; idx-=1
    else: subvars[idx] += cumsum; cumsum = 0
  submix = [subvars[j]*subvars[k] for j in range(nsubs) for k in range(j+1, nsubs)]
  cumsum, comb = 0, [n-2]
  for i in range(nsubs-2,-1,-1):
    comb.append(comb[-1] - max(1, subvars[i]-subvars[i+1]))
  comb = list(reversed(comb))
  subisolates = [-(b[0]*i+b[1]*i*(i-1)//2+b[2]+i*subvars[j]) for i,j in zip(comb, range(nsubs))]
  subchecks = [[-(b[0]*i+b[1]*i*(i-1)//2+b[2]+i*sum(subvars[k] for k in range(nsubs) if j&(1<<k)!=0)+
                  sum(subisolates[k] for k in range(nsubs) if j&(1<<k)!=0)+
                  sum(subvars[k]*subvars[l] for k in range(nsubs) for l in range(k+1, nsubs) if j&(1<<k)!=0 and j&(1<<l)!=0)
                  ) for j in range(1<<nsubs)] for i in range(1, n+1)]
  assert all(0 in x for x in subchecks) #existence/correctness
  zidxs = [x.index(0) for x in subchecks]
  sets = [{i+1 for i, z in enumerate(zidxs) if z&(1<<k)!=0} for k in range(nsubs)]
  #print(sets, b, subvars, subisolates, submix
  return sets, b, subvars, subisolates, submix
  #from scipy.optimize import linprog
  #import numpy as np
  #for sums in itertools.combinations(all_nary_count_combs(n), nsubs):
    #sums = [{1,2,3,4,5,6}, *sums]
    #if len(functools.reduce(operator.or_, [set(x) for x in sums])) < n-2: continue
    #a = [[i*int(i in sums[j]) for j in range(nsubs)] + [int(i in sums[j]) for j in range(nsubs)] + [int(i in sums[j])*int(i in sums[k]) for j in range(nsubs) for k in range(j+1, nsubs)] for i in range(1,n+1)]
    #b = [-((-n+1)*i+i*(i-1)//2+n*(n-1)//2) for i in range(1,n+1)]
    #res = linprog([1]*len(a[0]), A_ub=None, b_ub=None, A_eq=a, b_eq=b, bounds=(-(1<<n), 1<<n), method="highs", integrality=1)
    #if res.success: print(n, sums)
def test_cnf_linalg_solve():
  for i in range(3, 1024+1):
    print(i)
    _, b, subvars, subisolates, submix = cnf_linalg_solve(i)
#test_cnf_linalg_solve(); assert False
def all_nary_count_partitions(n, nsubs):
  import itertools
  #naive way
  #for combs in itertools.combinations([comb for i in range(1, n+1) for comb in itertools.combinations(range(n+1), i)], nsubs):
  #  combs = [set(comb) for comb in combs]
  #  if not valid_nary_combs(n, combs): continue
  #  yield [get_nary_count_func(comb) for comb in combs]    
  submixing = [1] + [nsubs] + [i*(i-1)//2 for i in range(2, nsubs+1)] #permutation groups
  totalpartitions = sum(submixing)
  for totalparts in (totalpartitions,): #range((n+1)//2, totalpartitions+1): #for non-all-overlapping
    for partition in subsets_k(list(range(1, n+1)), totalparts):
      partition = partition + [[]]*(totalpartitions-totalparts)
      for group in partition_groups(partition, submixing):
        subpartitions = [[] for _ in range(nsubs)]
        groupitem = iter(group)
        curgroup = next(groupitem)
        for i in range(1<<nsubs):
          for j in range(nsubs):
            if (i & (1<<j)) != 0:
              subpartitions[j].extend(curgroup[0])
          curgroup.pop(0)
          if len(curgroup) == 0 and i != (1<<nsubs)-1: curgroup = next(groupitem)
        #yield [get_nary_count_func(set(p)) for p in subpartitions]
        yield subpartitions
#print(len(list(all_nary_count_partitions(4, 1))))
#print([len(list(all_nary_count_partitions(i, 2))) for i in range(5, 8+1)])
#print(len(list(all_nary_count_partitions(9, 3))))
def all_nary_count_distributions(n, nsubs):
  import itertools, functools
  ndistros = (1<<nsubs)-1
  def sortfunc(a, b):
    l = min(len(a), len(b))
    al, bl = a[:l], b[:l]
    if al == bl: return 0 if len(a) == len(b) else (-1 if len(a) > len(b) else 1)
    return -1 if al < bl else 1
  order = list(sorted((x for i in range(1, nsubs+1) for x in itertools.combinations(range(nsubs), i)), key=functools.cmp_to_key(sortfunc)))
  for comb in itertools.combinations(range(ndistros), n-2-ndistros):
    counts = [2 if i in comb else 1 for i in range(ndistros)]
    print(counts)
    distros = [set() for _ in range(nsubs)]
    x = 1
    for idx, count in enumerate(counts):
      xset = {x} if count==1 else {x,x+1}; x+=count
      for group in order[idx]:
        distros[group] |= xset
    yield distros
#print(len(list(all_nary_count_partitions(13, 3))))
#print([nary_func_to_num(f, 5) for f in all_nary_count_funcs(5)])
def num_func_to_formula(val, n):
  import operator, functools
  from pyeda.inter import espresso_exprs, exprvar #, Not, And, Or, Xor, Xnor, Nand, Nor, Implies
  syms = [exprvar(chr(ord('a')+i)) for i in range(n-1,-1,-1)]
  f = functools.reduce(operator.or_, (functools.reduce(operator.and_, (syms[j] if (i & (1<<j)) != 0 else ~syms[j] for j in range(n))) for i in range(1<<n) if (val & (1<<i)) != 0))
  return espresso_exprs(f.to_dnf())[0]
  #from sympy import Symbol
  #from sympy.logic.boolalg import to_cnf, to_anf
  #syms = [Symbol(chr(ord('a')+i)) for i in range(n-1,-1,-1)]
  #f = functools.reduce(operator.or_, (functools.reduce(operator.and_, (syms[j] if (i & (1<<j)) != 0 else ~syms[j] for j in range(n))) for i in range(1<<n) if (val & (1<<i)) != 0))
  #print(f, to_cnf(f))
  #return f.simplify()
#print(list(num_func_to_formula(i, 2) for i in range(1, 16)))
def get_all_ternary_qubo_encoding():
  import numpy as np
  allbinarysubs2 = [lambda a,b,c: x(b, c, a) for x in allbinarysubs]
  allbinarysubs3 = [lambda a,b,c: x(a, c, b) for x in allbinarysubs]
  nosubencodings = [ilp_ideal_qubo_encoding(get_ternary_func(i), 3, [[]], None) for i in range(1<<8)]
  #print(sum(1 for x in nosubencodings if not x[0] is None))
  encodings = [nosubencodings[i] + ([None]*10*3 if not nosubencodings[i][0] is None else ilp_ideal_qubo_encoding(get_ternary_func(i), 3, [[x] for x in allbinarysubs+allbinarysubs2+allbinarysubs3], None)) for i in range(256)]
  best = [max((i for i in range(len(allbinarysubs)*3+1) if not z[i] is None), key=lambda x: np.count_nonzero(z[x][:-1] == 0), default=0) for z in encodings]
  return encodings, best
def ilp_ideal_qubo_encoding_allsubs_eff(f, n, nsubs):
  res = ilp_ideal_qubo_encoding_allsubs(f, n, nsubs)
  if res is None: return res
  opt = ilp_ideal_qubo_encoding(f, n, [[get_nary_func(i, n) for i in res[1]]])
  print(opt, res[0], res[1])#, [num_func_to_formula(i, n) for i in res[1]])
  return opt, res[1]
def max_coeff_ilp_qubo_encoding(n):
  import numpy as np
  maxsubs = (n-1).bit_length()-1
  mincoeffs, maxcoeffs = [None]*(maxsubs+1), [None]*(maxsubs+1)
  totals = {x: 0 for x in range(maxsubs+1)}
  for i in range(1<<(1<<n)):
    for nsubs in range(maxsubs+1):
      if nsubs == 0: res = ilp_ideal_qubo_encoding(get_nary_func(i, n), n, [[]], None, True)[0]
      else: res = ilp_ideal_qubo_encoding_allsubs(get_nary_func(i, n), n, nsubs)
      if not res is None:
        totals[nsubs] += 1
        res = ilp_ideal_qubo_encoding(get_nary_func(i, n), n, [[get_nary_func(res[1][sub], n) for sub in range(nsubs)]], None)[0]
        assert not res is None, (i, totals, mincoeffs, maxcoeffs)
        if mincoeffs[nsubs] is None: mincoeffs[nsubs] = res
        else: mincoeffs[nsubs] = np.minimum(mincoeffs[nsubs], res)
        if maxcoeffs[nsubs] is None: maxcoeffs[nsubs] = res
        else: maxcoeffs[nsubs] = np.maximum(maxcoeffs[nsubs], res)
        break
      assert nsubs != maxsubs or not res is None
  print(totals, mincoeffs, maxcoeffs)
def all_ternary_qubo_encoding():
  import itertools
  #print(ilp_ideal_qubo_encoding(lambda a, b: (1-a) & (1-b), 2, [[]], coeff=1))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c: (1-a) & (1-b) & (1-c), 3, [[]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g: (1-a) & (1-b) & (1-c) & (1-d) & (1-e) & (1-f) & (1-g), 7, [[]])); assert False
  #print(ilp_ideal_qubo_encoding(lambda a, b, c: a == b & c, 3, [[]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: a == b & c & d, 4, [[lambda a, b, c, d: get_nary_count_func({0})(b,c,d)]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e: a == b & c & d & e, 5, [[lambda a, b, c, d, e: get_nary_count_func({0,1})(b,c,d,e)]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: a == b & c & d & e & f, 6, [[lambda a, b, c, d, e, f: get_nary_count_func({0,1})(b,c,d,e,f), lambda a, b, c, d, e, f: get_nary_count_func({2})(b,c,d,e,f)]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g: a == b & c & d & e & f & g, 7, [[lambda a, b, c, d, e, f, g: get_nary_count_func({0,1})(b,c,d,e,f,g), lambda a, b, c, d, e, f, g: get_nary_count_func({2,3})(b,c,d,e,f,g)]]))  
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: a == b & c & d & e & f & g & h, 8, [[lambda a, b, c, d, e, f, g, h: get_nary_count_func({0,1,2})(b,c,d,e,f,g,h), lambda a, b, c, d, e, f, g, h: get_nary_count_func({0,1,3,4})(b,c,d,e,f,g,h)]]))  
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i: a == b & c & d & e & f & g & h & i, 9, [[lambda a, b, c, d, e, f, g, h, i: get_nary_count_func({0,1,2,3})(b,c,d,e,f,g,h,i), lambda a, b, c, d, e, f, g, h, i: get_nary_count_func({0,1,4,5})(b,c,d,e,f,g,h,i)]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i: a == b & c & d & e & f & g & h & i, 9, [[lambda a, b, c, d, e, f, g, h, i: get_nary_count_func({0,1,2,3})(b,c,d,e,f,g,h,i), lambda a, b, c, d, e, f, g, h, i: get_nary_count_func({0,1,4,5})(b,c,d,e,f,g,h,i)]]))
  #for i in range(1<<8):
  #  s1 = {x for x in range(8) if (1<<x)&i!=0}
  #  for j in range(i+1,1<<8):
  #    s2 = {x for x in range(8) if (1<<x)&j!=0}
  #    res = ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: a == b & c & d & e & f & g & h, 8, [[lambda a, b, c, d, e, f, g, h: get_nary_count_func(s1)(b,c,d,e,f,g,h), lambda a, b, c, d, e, f, g, h: get_nary_count_func(s2)(b,c,d,e,f,g,h)]])
  #    if not res[0] is None: print(s1,s2,res)
  #  print(i,s1)
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f, g, h: a == b & c & d & e & f & g & h, 8, 1, knownsubs=[get_nary_count_func({2,3,6})]))
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d: a == b & c & d, 4, 1)) #[32888]
  #print(iqp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e: a == b & c & d & e, 5, 1)) #[32488] [279]
  #print(iqp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f: a == b & c & d & e & f, 6, 2)) #[18290558, 1]
  #print(iqp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f, g: a == b & c & d & e & f & g, 7, 2)) #[4295033111, 170141183460469231731765861064713862889]
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j: a | b | c | d | e | f | g | h | i | j, 10, [[get_nary_count_func({1,2,3,4,5}), get_nary_count_func({1,2,3,6,7}), get_nary_count_func({1,2,4,6,8})]], None)) #2,1,1,1,1,1,1
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l, m: a | b | c | d | e | f | g | h | i | j | k | l | m, 13, [[get_nary_count_func(x) for x in comb] for comb in all_nary_count_distributions(13, 3)], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k: a | b | c | d | e | f | g | h | i | j | k, 11, [[get_nary_count_func({1,2,3,4,5}), get_nary_count_func({1,2,3,6,7,8}), get_nary_count_func({1,2,4,6,9})]], None, True)) #2,1,1,1,1,2,1
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k: a | b | c | d | e | f | g | h | i | j | k, 11, [[get_nary_count_func({1,2,3,4,5}), get_nary_count_func({1,2,3,6,7,8}), get_nary_count_func({1,2,4,6,7,9})]], None, True)) #2,1,1,1,2,1,1
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l: a | b | c | d | e | f | g | h | i | j | k | l, 12, [[get_nary_count_func({1,2,3,4,5,6}), get_nary_count_func({1,2,3,7,8,9}), get_nary_count_func({1,2,4,5,7,8,10})]], None, True)) #2,1,2,1,2,1,1
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l: a | b | c | d | e | f | g | h | i | j | k | l, 12, [[get_nary_count_func({1,2,3,4,5,6}), get_nary_count_func({1,2,3,7,8,9}), get_nary_count_func({1,2,4,7,10})]], None, True)) #2,1,1,2,1,2,1
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l, m: a | b | c | d | e | f | g | h | i | j | k | l | m, 13, [[get_nary_count_func({1,2,3,4,5,6}), get_nary_count_func({1,2,3,4,7,8,9}), get_nary_count_func({1,2,5,6,7,10,11})]], None, True))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l, m: a | b | c | d | e | f | g | h | i | j | k | l | m, 13, [[get_nary_count_func({1,2,3,4,5,6,7}), get_nary_count_func({1,2,3,4,8,9}), get_nary_count_func({1,2,5,8,10,11})]], None, True)) #2,2,1,2,1,1,2
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l, m, n: a | b | c | d | e | f | g | h | i | j | k | l | m | n, 14, [[get_nary_count_func({1,2,3,4,5,6,7}), get_nary_count_func({1,2,3,4,8,9,10}), get_nary_count_func({1,2,5,6,8,11,12})]], None, True)) #2,2,2,1,1,2,2
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l, m, n, o: a | b | c | d | e | f | g | h | i | j | k | l | m | n | o, 15, [[get_nary_count_func({1,2,3,4,5,6,7,8}), get_nary_count_func({1,2,3,4,9,10,11}), get_nary_count_func({1,2,5,6,9,12,13})]], None, True)) #2,2,2,2,1,2,2
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l, m, n, o: a | b | c | d | e | f | g | h | i | j | k | l | m | n | o, 15, [[get_nary_count_func({1,2,3,4,5,6,7}), get_nary_count_func({1,2,3,4,8,9,10,11}), get_nary_count_func({1,2,5,6,8,9,12,13})]], None, True)) #2,2,2,1,2,2,2
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p: a | b | c | d | e | f | g | h | i | j | k | l | m | n | o | p, 16, [[get_nary_count_func({1,2,3,4,5,6,7,8}), get_nary_count_func({1,2,3,4,9,10,11,12}), get_nary_count_func({1,2,5,6,9,10,13,14})]], None, True)) #2,2,2,2,2,2,2
  #print(list(filter(lambda x: not x is None, ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j, k, l, m: a | b | c | d | e | f | g | h | i | j | k | l | m, 13, [[get_nary_count_func({1,2,3,4,5,6}), get_nary_count_func({1,2} | set(comb1)), get_nary_count_func({1,2} | set(comb2))] for comb1 in itertools.combinations(range(3, 11+1), 4) for comb2 in itertools.combinations(range(3, 11+1), 5)], None, True))))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i: a | b | c | d | e | f | g | h | i, 9, [[get_nary_count_func({1,2,3,4,5,6}), get_nary_count_func({1,2,7,8,9,10}), get_nary_count_func({1,3,5,7,9,11,12,13})]], None))
  #print(list(filter(lambda x: not x is None, ilp_ideal_qubo_encoding(lambda a, b, c, d: a | b | c | d, 4, [[get_nary_count_func(x)] for x in all_nary_count_combs(4)], None, True))))
  
  #find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d: a|b|c|d, 4, 1); assert False
  #find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f: a & b ^ c & d ^ e & f, 6, 2); assert False
  #find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f: a|b|c|d|e|f, 6, 2); assert False
  #find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f: (a & b ^ c & d ^ e & f), 6, 2, knownsubs=[get_nary_count_func({2,3})]); assert False
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f: get_nary_count_func({1,3})(a&b,c&d,e&f), 6, 2)); assert False
  #print(iqp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f: get_nary_count_func({1,3})(a&b,c&d,e&f), 6, 2)); assert False
  #ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f, g, h: get_nary_count_func({1,3})(a&b,c&d,e&f,g&h), 8, 3, knownsubs=[lambda a, b, c, d, e, f, g, h: get_nary_count_func({2,3})(a&b,c&d,e&f,g&h), lambda a, b, c, d, e, f, g, h: get_nary_count_func({4})(a&b,c&d,e&f,g&h)]); assert False
  #find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f, g, h: a & b ^ c & d ^ e & f ^ g & h, 8, 1, knownsubs=[get_nary_count_func({2,3}), get_nary_count_func({4})]); assert False
  #print(ilp_ideal_qubo_encoding_allsubs_eff(lambda a, b, c, d: a ^ b ^ c ^ d, 4, 1))
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e: a ^ b ^ c ^ d ^ e, 5, 2)); assert False
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e: a & b ^ c & d ^ e, 5, 2, knownsubs=[lambda a, b, c, d, e: get_nary_count_func({2,3})(a&b,c&d,e)]))
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e: (a & b) ^ (c & d) ^ e, 5, 2))
  #print(iqp_ideal_qubo_encoding_allsubs(lambda a, b, c, d: a|b|c|d, 4, 1))
  #print(iqp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e: (a & b) ^ (c & d) ^ e, 5, 2))
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f: (a & b) | (c & d) | (e & f), 6, 2))
  #print(nary_func_to_num(lambda a, b, c, d, e: get_nary_count_func({1})(a&b,c&d,e), 5))
  #print(nary_func_to_num(lambda a,b,c,d,e,f: a & b ^ c & d ^ e & f, 6)); assert False
  #print(num_func_to_formula(nary_func_to_num(lambda a,b,c,d,e,f: a & b ^ c & d ^ e & f, 6), 6))
  #print(nary_func_to_num(lambda a, b, c, d: a == (b ^ c ^ d), 4))  
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: (a == get_nary_count_func({2})(c,d)) & (b == get_nary_count_func({1})(c,d)), 4, [[]])) #half adder
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e: (a == get_nary_count_func({2,3})(c,d,e)) & (b == get_nary_count_func({1,3})(c,d,e)), 5, [[]])) #full adder
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g: (a == get_nary_count_func({4})(d,e,f,g)) & (b == get_nary_count_func({2,3})(d,e,f,g)) & (c == get_nary_count_func({1,3})(d,e,f,g)), 7, [[]])) #4-adder
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: (a == get_nary_count_func({4,5})(d,e,f,g,h)) & (b == get_nary_count_func({2,3})(d,e,f,g,h)) & (c == get_nary_count_func({1,3,5})(d,e,f,g,h)), 8, [[]])) #5-adder
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i: (a == get_nary_count_func({4,5,6})(d,e,f,g,h,i)) & (b == get_nary_count_func({2,3,6})(d,e,f,g,h,i)) & (c == get_nary_count_func({1,3,5})(d,e,f,g,h,i)), 9, [[]])) #6-adder
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h, i, j: (a == get_nary_count_func({4,5,6,7})(d,e,f,g,h,i,j)) & (b == get_nary_count_func({2,3,6,7})(d,e,f,g,h,i,j)) & (c == get_nary_count_func({1,3,5,7})(d,e,f,g,h,i,j)), 10, [[]])) #7-adder
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e: (a == get_nary_count_func({2})(b&c,d&e)) & get_nary_count_func({1})(b&c,d&e), 5, [[]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e: (a == get_nary_count_func({2})(b&c,d&e)) & (1-get_nary_count_func({1})(b&c,d&e)), 5, [[]]))
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f: (a == get_nary_count_func({2})(c&d,e&f)) & (b == get_nary_count_func({1})(c&d,e&f)), 6, 1))
  #num_func_to_formula(0x8880000, 6); num_func_to_formula(0x70000000, 6)
  #print(iqp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f, g: (a == get_nary_count_func({2,3})(b&c,d&e,f&g)) & get_nary_count_func({1,3})(b&c,d&e,f&g), 7, 1))
  #print(iqp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f, g: (a == get_nary_count_func({2,3})(b&c,d&e,f&g)) & (1-get_nary_count_func({1,3})(b&c,d&e,f&g)), 7, 1))
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f, g, h: (a == get_nary_count_func({2,3})(c&d,e&f,g&h)) & (b == get_nary_count_func({1,3})(c&d,e&f,g&h)), 8, 1))
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f, g, h: (a == get_nary_count_func({2,3})(c&d,e&f,g&h)) & (b == get_nary_count_func({1,3})(c&d,e&f,g&h)), 8, 1))
  print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f, g, h, i, j, k: (a == get_nary_count_func({4})(d&e,f&g,h&i,j&k)) & (b == get_nary_count_func({2,3})(d&e,f&g,h&i,j&k)) & (c == get_nary_count_func({1,3})(d&e,f&g,h&i,j&k)), 11, 1)); assert False
  num_func_to_formula(0x100000000000000001111000100010000000011101110111, 8)
  num_func_to_formula(0x800000000000000078880000000000000777, 8)
  assert False
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f: (a == (c & d & e & f)) & (b == (c & d ^ e & f)), 6, 1)); assert False
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, c, d, e, f, g, h: (a == get_nary_count_func({2,3})(c&d,e&f,g&h)) & get_nary_count_func({1,3})(c&d,e&f,g&h), 7, 1)); assert False
  print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f, g, h, i, j, k: (a == get_nary_count_func({4})(d&e,f&g,h&i,j&k)) & (b == get_nary_count_func({2,3})(d&e,f&g,h&i,j&k)) & (c == get_nary_count_func({1,3})(d&e,f&g,h&i,j&k)), 11, 2, knownsubs=[get_nary_func(0x00000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000011110001000100010000000000000001000000000000000100000000000000000000111011101110111100010001000011110001000100001111000100010000000000000000000000001110111011100000111011101110000011101110111, 11)])); assert False
  print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f, g, h: (a == get_nary_count_func({2,3})(c&d,e&f,g&h)) & (b == get_nary_count_func({1,3})(c&d,e&f,g&h)), 8, 1, knownsubs=[get_nary_func(0x100000000000000001111000100010000000011101110111, 8)])); assert False #0x100000000000000001111000100010000000011101110111 0x800000000000000078880000000000000777
  print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e: (a == (b & c | c & d | b & d)) & (e == (b & c | c & d | b & d)), 5, 1))
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: get_nary_count_func({3})(a&b,c&d,e&f), 6, [[]]))
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: get_nary_count_func({2})(a&b,c&d,e&f), 6, [[]]))
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: get_nary_count_func({0})(a&b,c&d,e&f), 6, [[]]))
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g: get_nary_count_func({2,3})(a&b,c&d,e&f)==g, 7, [[lambda a, b, c, d, e, f, g: get_nary_count_func({1})(a&b,c&d,e&f)]]))
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: get_nary_count_func({4})(a&b,c&d,e&f,g&h), 8, [[]]))
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: get_nary_count_func({3})(a&b,c&d,e&f,g&h), 8, [[]]))
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: get_nary_count_func({0})(a&b,c&d,e&f,g&h), 8, [[]]))
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: get_nary_count_func({2,3,4})(a&b,c&d,e&f,g&h), 8, [[lambda a, b, c, d, e, f, g, h: get_nary_count_func({0,1})(a&b,c&d,e&f,g&h), lambda a, b, c, d, e, f, g, h: get_nary_count_func({4})(a&b,c&d,e&f,g&h)]]))
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f: get_nary_count_func({1})(a&b,c&d,e&f), 6, 1))
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f, g, h: get_nary_count_func({1,2})(a&b,c&d,e&f,g&h), 8, 1))
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f, g, h: get_nary_count_func({1})(a&b,c&d,e&f,g&h), 8, 1))
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f, g, h: get_nary_count_func({2})(a&b,c&d,e&f,g&h), 8, 1))
  #print(find_ideal_qubo_encoding_allsubs_tree(lambda a, b, c, d, e, f: get_nary_count_func({2,3})(a&b,c&d,e&f), 6, 1))
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f: a | b | c | d | e | f, 6, 2))
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f: a & b ^ c & d ^ e & f, 6, 2, knownsubs=[lambda a,b,c,d,e,f: (a & b ^ c & d) & ~(e & f)]))
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f, g, h: a & b ^ c & d ^ e & f & g & h, 8, 3, knownsubs=[get_nary_count_func({2,3}), get_nary_count_func({4})]))
  #print(iqp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f: a & b ^ c & d ^ e & f, 6, 2)); assert False #8685332621019877448, 8685614095996583936
  #print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e: a & b ^ c & d == e, 5, 2))
  #a&b^c&d==a&b&(~c|~d)|(~a|~b)&c&d
  #a&b&(~c|~d)&e&f|(~a|~b)&c&d&e&f|a&b|(~c|~d)&(~a|~b)|c&d&(~e|~f)
  #ab+cd+ef-s-3t+4
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: a & b ^ c & d ^ e & f, 6, [[lambda a, b, c, d, e, f: get_nary_count_func({1})(a&b,c&d,e&f), lambda a, b, c, d, e, f: get_nary_count_func({3})(a&b,c&d,e&f)]]))
  print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: a & b ^ c & d ^ e & f ^ g & h, 8, [[lambda a, b, c, d, e, f, g, h: a & b & (~c|~d) & (~e|~f) & (~g|~h), lambda a, b, c, d, e, f, g, h: (~a|~b) & c&d & (~e|~f) & (~g|~h), lambda a, b, c, d, e, f, g, h: (~a|~b) & (~c|~d) & e&f & (~g|~h)]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: a & b ^ c & d ^ e & f, 6, [[get_nary_func(8685332621019877448, 6), lambda a, b, c, d, e, f: get_nary_func(8685614095996583936,6)(a,b, c,d, e,f)]]))  
  for i in range(1<<8):
    for j in range(1<<8): #range(i+1,1<<8):
      if ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: a & b ^ c & d ^ e & f, 6, [[lambda a, b, c, d, e, f: get_nary_func(i, 3)(a, c, e), lambda a, b, c, d, e, f: get_nary_func(j, 3)(b, d, f)]]) != [None]:
        print(i,j)
    print(i, "complete")
  #for i in range(1<<8):
  #  print(ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d, e, f: a & b ^ c & d ^ e & f, 6, 2, knownsubs=[lambda a, b, c, d, e, f: get_nary_func(i, 3)(a&b,c&d,e&f)]))  
  #max_coeff_ilp_qubo_encoding(3) #{0: 226, 1: 30} [array([-3, -3, -3, -2, -2, -2,  0]), array([-3, -3, -3, -2, -2, -2, -4, -4, -4, -4,  0])] [array([3, 3, 3, 2, 2, 2, 4]), array([3, 3, 3, 2, 2, 2, 4, 3, 4, 4, 5])]
  #max_coeff_ilp_qubo_encoding(4) #partial {0: 14832, 1: 24330}, [array([-8, -8, -8, -8, -4, -4, -4, -4, -4, -4,  0]), array([-16, -15, -12, -13,  -8,  -8,  -7,  -9,  -8, -11, -11, -12, -12, -12, -18,   0])], [array([8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 9]), array([13, 13, 12, 13, 10, 10, 10, 12,  9, 10, 10, 10, 10,  9, 17, 22])])
  #nosubencodings4 = [ilp_ideal_qubo_encoding(get_quaternary_func(i), 4, [[]], None) for i in range(1<<16)]
  #print(sum(1 for x in nosubencodings4 if not x[0] is None)) #20298
  #for i in range(1<<16):
  #  if nosubencodings4[i][0] is None:
  #    nosubencodings4[i][0] = ilp_ideal_qubo_encoding_allsubs(get_quaternary_func(i), 4, 1)
  #  assert not nosubencodings4[i][0] is None
  #encodings4 = [ilp_ideal_qubo_encoding(get_quaternary_func(i), 4, [[lambda a, b, c, d: a or b]], None) for i in range(1<<16)]
  #print(sum(1 for x in encodings4 if not x[0] is None)) #38338  
  #print(ilp_ideal_qubo_encoding(lambda a, b, c: a ^ b ^ c, 3, [[get_nary_count_func({2,3})]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: a ^ b ^ c ^ d, 4, [[get_nary_count_func({2,3}), get_nary_count_func({4})]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e: a ^ b ^ c ^ d ^ e, 5, [[get_nary_count_func({2,3}), get_nary_count_func({4,5})]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: a ^ b ^ c ^ d ^ e ^ f, 6, [[get_nary_count_func({2,3,6}), get_nary_count_func({4,5,6})]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g: a ^ b ^ c ^ d ^ e ^ f ^ g, 7, [[get_nary_count_func({2,3,6,7}), get_nary_count_func({4,5,6,7})]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: a ^ b ^ c ^ d ^ e ^ f ^ g ^ h, 8, [[get_nary_count_func({2,3,6,7}), get_nary_count_func({4,5,6,7}), get_nary_count_func({8})]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: a & b ^ c & d, 4, [[]], None)) #sum of half adder of quadratic terms
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: a & b & c & d, 4, [[]], None)) #carry of half adder of quadratic terms #-ac-bd+2
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e: a & b ^ c & d ^ e, 5, [[lambda a, b, c, d, e: get_nary_count_func({2,3})(a&b,c&d,e), lambda a, b, c, d, e: a & b & (~c | ~d) & ~e]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: a & b ^ c & d ^ e & f, 6, [[lambda a, b, c, d, e, f: get_nary_count_func({2,3})(a&b,c&d,e&f), lambda a, b, c, d, e, f: get_nary_count_func({1,3})(a&b,c&d,e&f)]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: a & b ^ c & d ^ e & f ^ g & h, 8, [[lambda a, b, c, d, e, f, g, h: get_nary_count_func({4})(a&b,c&d,e&f,g&h), lambda a, b, c, d, e, f, g, h: get_nary_count_func({2,3})(a&b,c&d,e&f,g&h), lambda a, b, c, d, e, f, g, h: get_nary_count_func({1,3})(a&b,c&d,e&f,g&h)]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: a & b & c & d | (e & f & (a & b ^ c & d)), 6, [[]], None)) #carry of full adder does not require substitution
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: get_nary_count_func({2,3})(a&b, c&d, e&f), 6, [[]], None))
  #print(list(filter(lambda x: not x is None, ilp_ideal_qubo_encoding(lambda a, b, c, d: a & b ^ c & d, 4, [[get_quaternary_func(i)] for i in range(1<<16)], None, True))))
  #print(list(filter(lambda x: not x is None, ilp_ideal_qubo_encoding(lambda a, b, c, d, e: (a & b) ^ (c & d) ^ e, 5, [[lambda a, b, c, d, e: get_quaternary_func(i)(a,b,c,d)] for i in range(1<<16)], None, True))))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: (a & b) ^ (c & d), 4, [[]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: (a & b) & (c & d), 4, [[]]))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: (a & b) ^ (c & d) ^ (e & f) ^ (g & h), 8, [[lambda a, b, c, d, e, f, g, h: (a | b) ^ (c | d), lambda a, b, c, d, e, f, g, h: (e | f) ^ (g | h)]])); assert False
  #print(list(filter(lambda x: not x is None, ilp_ideal_qubo_encoding(lambda a, b, c, d: a | b | c | d, 4, [[lambda a, b, c, d: get_quaternary_func(i)(a,b,c,d)] for i in range(1<<16)], None, True))))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: (1-a) or (1-b) or (1-c) or (1-d), 4, [[lambda a, b, c, d: get_ternary_func(i)(a,b,c)] for i in range(256)], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: a or b or c or d, 4,
  #                              [[lambda a, b, c, d: a ^ b ^ c ^ d, lambda a, b, c, d: a&b&c&d, lambda a, b, c, d: (a & (b | c | d) | b & (c | d) | c & d) & (1-(a&b&c&d))]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: a or b or c or d, 4,
  #                              [[lambda a, b, c, d: a or b, lambda a, b, c, d: c or d]], None))
  #print(list(filter(lambda x: not x is None, ilp_ideal_qubo_encoding(lambda a, b, c, d, e: a or b or c or d or e, 5,
  #                              [[lambda a, b, c, d, e: get_ternary_func(i)(a,b,c), lambda a, b, c, d, e: get_quaternary_func(j)(b, c, d, e)] for i in range(1, 256) for j in range(1<<16)], None, True))))
  #print(list(filter(lambda x: not x is None, ilp_ideal_qubo_encoding(lambda a, b, c, d, e: a or b or c or d or e, 5,
  #                              [[lambda a, b, c, d, e: a & b & c & d & e, lambda a, b, c, d, e: get_quinary_func(i)(a, b, c, d, e)] for i in range(1<<32)], None, True))))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e: a or b or c or d or e, 5,
  #                              [[lambda a, b, c, d, e: a ^ b ^ c ^ d ^ e, lambda a, b, c, d, e: a&b&c&(d|e)|b&c&d&e, lambda a, b, c, d, e: (a & (b | c | d | e) | b & (c | d | e) | c & (d | e) | d & e) & (1-(a&b&c&(d|e)|b&c&d&e))]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e: a or b or c or d or e, 5,
  #                              [[lambda a, b, c, d, e: a or b, lambda a, b, c, d, e: c or d, lambda a, b, c, d, e: a or b or e]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: a or b or c or d or e or f, 6,
  #                              [[lambda a, b, c, d, e, f: a or b, lambda a, b, c, d, e, f: c or d, lambda a, b, c, d, e, f: a or b or c or d, lambda a, b, c, d, e, f: e or f]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: a or b or c or d or e or f or g or h, 8,
  #                              [[lambda a, b, c, d, e, f, g, h: a or b, lambda a, b, c, d, e, f, g, h: c or d, lambda a, b, c, d, e, f, g, h: e or f, lambda a, b, c, d, e, f, g, h: g or h,
  #                              lambda a, b, c, d, e, f, g, h: a or b or c or d, lambda a, b, c, d, e, f, g, h: e or f or g or h]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f, g, h: a or b or c or d or e or f or g or h, 8,
  #                              [[lambda a, b, c, d, e, f, g, h: a or b, lambda a, b, c, d, e, f, g, h: c or d, lambda a, b, c, d, e, f, g, h: e or f, lambda a, b, c, d, e, f, g, h: g or h,
  #                              lambda a, b, c, d, e, f, g, h: a or b or c or d, lambda a, b, c, d, e, f, g, h: e or f or g or h]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: (1-a) or (1-b) or (1-c) or (1-d), 4, [[lambda a, b, c, d: a and b, lambda a, b, c, d: c and d]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: int(a == (b ^ c ^ d)), 4, [[lambda a, b, c, d: a & b, lambda a, b, c, d: c & d]], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d: int(a == (b ^ c ^ d)), 4, [[lambda a, b, c, d: get_ternary_func(i)(b,c,d)] for i in range(256)], None))
  #print(ilp_ideal_qubo_encoding(lambda a, b, c, d, e, f: int(((a & b) ^ (c & d) ^ (e & f)) == 0), 6, [[lambda a, b, c, d, e, f: get_ternary_func(i)(a,c,e)] for i in range(256)], None))
  import numpy as np
  encodings, best = all_ternary_qubo
  print(best)
  print([sum(1 for z in encodings if not z[i] is None) for i in range(len(allbinarysubs)*3+1)])
  #badidxs = [i for i, x in enumerate(encodings) if x[1] is None]
  syms = all_ternary_funcs()
  symnames, subnames = ("a", "b", "c", "d", "e", "f", "g", "h"), ("s", "t", "u")
  binopnames = ("∧", "⊼", "∨", "⊽", "⊕ ", "↔", "⇏ ", "→", "⇍ ", "←")
  negsymltx = "\\neg "
  binopnamesltx = ("\\wedge ", "\\barwedge ", "\\vee ", "\\barvee ", "\\oplus ", "\\leftrightarrow ", "\\nrightarrow ", "\\implies ", "\\nleftarrow ", "\\leftarrow ")
  binopnames, negsym = binopnamesltx, negsymltx
  def get_ltx_sub(enc, i):
    return "" if (enc[i][6:10] == 0).all() else ("$" + symnames[0] + binopnames[i-1] + symnames[1] + "$")
  def get_symnames(n, nsubs):
    return [symnames[i] for i in range(n)] + [symnames[i]+symnames[j] for i in range(n) for j in range(i+1, n)] + [symnames[j]+subnames[i] for i in range(nsubs) for j in range(n)] + [subnames[i] for i in range(nsubs)] + [subnames[i]+subnames[j] for i in range(nsubs) for j in range(i+1, nsubs)] + ['']
  def get_ltx_form(c, labels):
    return "$" + ''.join(('{0:+}'.format(i)[0:1 if abs(i)==1 and lbl!='' else None] + lbl) if i!=0 else "" for i, lbl in zip(c, labels)) + "$"
  for i in range(256):
    if best[i] != 0:
      print("$" + str(i) + "$", "&", "$" + syms[i] + "$", "&", get_ltx_sub(encodings[i], best[i]), "&",
            get_ltx_form(encodings[i][best[i]], get_symnames(3, 0 if best[i] == 0 else 1)), "\\\\")

  import functools, operator
  for negationcount in range(4+1):
    print(
      "$" + binopnames[2].join([(negsymltx if 4-1-i<negationcount else "") + symnames[i] for i in range(4)]) + "$", "&",
      "$\\text{count}(" + ",".join([(negsymltx if 4-1-i<negationcount else "") + symnames[i] for i in range(4)]) + ")$",
      "&", get_ltx_form(ilp_ideal_qubo_encoding(lambda *args: functools.reduce(operator.or_, [1-x if 4-1-i<negationcount else x for i, x in enumerate(args)]), 4, [[lambda *args: get_nary_count_func(x)(*[1-z if 4-1-i<negationcount else z for i, z in enumerate(args)]) for x in cnf_linalg_solve(4)[0]]], None)[0], get_symnames(4,1)), "\\\\")
    
  for i in range(4, 16+1):
    covsets, coeffvar, coeffsubvar, coeffsub, coeffsubmix = cnf_linalg_solve(i)
    print('$' + str(i) + '$-CNF', '&', 'Count sets: $' + ','.join('\\{' + ','.join(str(z) for z in x) + '\\}' for x in covsets) + '$',
          "\\\\", '\\multicolumn{2}{c}{' + '$(' + str(coeffvar[0]) + ' + ' + '+'.join([str(x) + "s_" + str(i) for i, x in enumerate(coeffsubvar)]) + ")\sum\limits_{i \in [1..n]} x_i + " + (str(coeffvar[1]) if coeffvar[1]!=1 else "") + "\sum\limits_{i,j \in [1..n], i<j} x_ix_j + " +
          "+".join(str(x) + "s_" + str(i) for i, x in enumerate(coeffsub)) + " + " +
          '+'.join(str(x) + "s_" + str(ij[0]) + "s_" + str(ij[1]) for x, ij in zip(coeffsubmix, ((i, j) for i in range(len(coeffsub)) for j in range(i+1, len(coeffsub))))) + " + " +
          str(coeffvar[2]) + '$' + '}', "\\\\")
    #print(ilp_ideal_qubo_encoding(lambda *args: functools.reduce(operator.or_, args), i, [[get_nary_count_func(x) for x in cnf_linalg_solve(i)[0]]], None, i>=12))

  #print(badidxs, [syms[idx] for idx in badidxs])
  #print([[x, encodings[i][x]] for i, x in enumerate(best)])
  #validates 2-SAT encodings:
  #binary_ops = [lambda a, b: a or b, lambda a, b: a or (1-b), lambda a, b: (1-a) or (1-b)]
  #idxs = [sum(op(a,b)<<(4*a+2*b+c) for a,b,c in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))) for op in binary_ops]
  #print([encodings[idx][best[idx]] for idx in idxs])
  #ab==s -> 3s+ab-2as-2bs
  #abc -> -b-ac+2
  ops = [lambda a, b, c: int((a and b) == c), lambda a, b, c: (1-a) or (1-b) or (1-c)]
  negsubst = ilp_ideal_qubo_encoding(ops[1], 3, [[x] for x in allbinarysubs], -1)
  bestnegsubst = max((i for i in range(len(allbinarysubs)) if not negsubst[i] is None), key=lambda x: np.count_nonzero(negsubst[x][:-1] == 0), default=0)
  print("Optimal negative substitution, normal substitution, optimal positive substitution")
  print("&", get_ltx_sub(negsubst, bestnegsubst), "&",
          get_ltx_form(negsubst[bestnegsubst], get_symnames(3, 1)), "\\\\")
  idxs = [nary_func_to_num(op, 3) for op in ops]
  for idx in idxs:
    print("&", get_ltx_sub(encodings[idx], best[idx]), "&",
          get_ltx_form(encodings[idx][best[idx]], get_symnames(3, 0 if best[idx] == 0 else 1)), "\\\\")

  print("Optimal 2-CNF Encodings")
  ops = [lambda a, b, c: a or b, lambda a, b, c: a or (1-b), lambda a, b, c: (1-a) or b, lambda a, b, c: (1-a) or (1-b)]
  negidxs = [(0, 0), (0, 1), (1, 0), (1, 1)]
  idxs=[nary_func_to_num(op, 3) for op in ops]
  for i, idx in enumerate(idxs):
    print("$" + (negsym if negidxs[i][0] else "") + symnames[0] + binopnamesltx[2] + (negsym if negidxs[i][1] else "") + symnames[1] + "$", "&",
          get_ltx_form(encodings[idx][best[idx]], get_symnames(3, 0 if best[idx] == 0 else 1)), "\\\\")
  print("Optimal Overall 3-CNF Encodings")
  idxs=[nary_func_to_num(op, 3) for op in allternarysat]
  negidxs = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
  for i, idx in enumerate(idxs):
    print("$" + (negsym if negidxs[i][0] else "") + symnames[0] + binopnamesltx[2] + (negsym if negidxs[i][1] else "") + symnames[1] + binopnamesltx[2] + (negsym if negidxs[i][2] else "") + symnames[2] + "$", "&",
          get_ltx_sub(encodings[idx], best[idx]), "&",
          get_ltx_form(encodings[idx][best[idx]], get_symnames(3, 1)), "\\\\")
  print("Best on Average Fixed Substitution 3-CNF Encodings")
  overallmax = [(i, np.sum([np.count_nonzero(encodings[idx][i][:-1] == 0) for idx in idxs])) for i in range(len(allbinarysubs)+1) if all(not encodings[idx][i] is None for idx in idxs)]
  overallidx = max(overallmax, key=lambda x: x[1])[0]
  for i, idx in enumerate(idxs):
    print("$" + (negsym if negidxs[i][0] else "") + symnames[0] + binopnamesltx[2] + (negsym if negidxs[i][1] else "") + symnames[1] + binopnamesltx[2] + (negsym if negidxs[i][2] else "") + symnames[2] + "$", "&",
          get_ltx_sub(encodings[idx], overallidx), "&",
          get_ltx_form(encodings[idx][overallidx], get_symnames(3, 1)), "\\\\")

all_ternary_qubo = get_all_ternary_qubo_encoding()
def check_encodings():
  def check(validsub, correct, computed): return (correct == 0) == (computed == 0) if validsub else correct <= computed
  opt3, opt2m1_1, opt2m1_2, opt2m1_3, opt1m2_1, opt1m2_2, opt1m2_3, optm3 = [all_ternary_qubo[0][nary_func_to_num(ternsat, 3)][1:len(allbinarysubs)+1] for ternsat in allternarysat]
  for optidx in range(0, len(allbinarysubs)):
    for i, C in enumerate((opt3[optidx], opt2m1_1[optidx], opt2m1_2[optidx], opt2m1_3[optidx], opt1m2_1[optidx], opt1m2_2[optidx], opt1m2_3[optidx], optm3[optidx])):
      if C is None: continue
      c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = C.tolist()
      assert all([check(s == allbinarysubs[optidx](x_i, x_j, x_k), 1-allternarysat[i](x_i, x_j, x_k), c_0*x_i+c_1*x_j+c_2*x_k+c_3*x_i*x_j+c_4*x_i*x_k+c_5*x_j*x_k+c_6*x_i*s+c_7*x_j*s+c_8*x_k*s+c_9*s+c_10) for x_i, x_j, x_k, s in [(x_i,x_j,x_k,s) for s in (0, 1) for x_i, x_j, x_k in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))]])
#all_ternary_qubo_encoding(); assert False
#check_encodings(); assert False
#print(find_ideal_qubo_encoding(lambda a, b, c: a or b or c, allbinarysubs))
#print(find_ideal_qubo_encoding(lambda a, b, c: a or b or (1-c), allbinarysubs + altbinarysubs))
#print(find_ideal_qubo_encoding(lambda a, b, c: a or (1-b) or (1-c), allbinarysubs + altbinarysubs))
#print(find_ideal_qubo_encoding(lambda a, b, c: (1-a) or (1-b) or (1-c), allbinarysubs))
#print(find_ideal_qubo_encoding(lambda a, b, c: 1-(a ^ (b and c))))
#https://arxiv.org/pdf/2302.03536.pdf
def max_three_sat_qubo(clauses, key, varidx, idx, varmap, bqm, const, submap, nusslein=False):
  import numpy as np
  #def check(validsub, correct, computed): return (correct == 0) == (computed == 0) if validsub else correct <= computed
  if nusslein:
    #Nusslein - a, b, c: [check(s == (x_i or x_j), 1-(x_i or x_j or x_k), 1+2*x_i*x_j-2*x_i*s-2*x_j*s-x_k+x_k*s+s) for x_i, x_j, x_k, s in [(x_i,x_j,x_k,gen(x_i or x_j)) for gen in (lambda z: z, lambda z: 1-z) for x_i, x_j, x_k in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))]]
    #Nusslein - a, b, ~c: [2*x_i*x_j-2*x_i*s-2*x_j*s+x_k-x_k*s+2*s for x_i, x_j, x_k, s in [(x_i,x_j,x_k,gen(x_i or x_j)) for gen in (lambda z: z, lambda z: 1-z) for x_i, x_j, x_k in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))]]
    #Nusslein - a, ~b, ~c: [2*x_i-2*x_i*x_j-2*x_i*s+2*x_j*s+x_k-x_k*s for x_i, x_j, x_k, s in [(x_i,x_j,x_k,gen(x_i or (1-x_j))) for gen in (lambda z: z, lambda z: 1-z) for x_i, x_j, x_k in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))]]
    #Nusslein - ~a, ~b, ~c: [1-x_i+x_i*x_j+x_i*x_k+x_i*s-x_j+x_j*x_k+x_j*s-x_k+x_k*s-s for x_i, x_j, x_k, s in [(x_i,x_j,x_k,gen((1-x_i) and (1-x_j) and (1-x_k))) for gen in (lambda z: z, lambda z: 1-z) for x_i, x_j, x_k in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))]]
    o3 = [0,0,-1,2,0,0,-2,-2,1,1,1] #a or b
    o2m1 = [0,0,1,2,0,0,-2,-2,-1,2,0] #a or b
    o1m2 = [2,0,1,-2,0,0,-2,2,-1,0,0] #a or not b
    om3 = [-1,-1,-1,1,1,1,1,1,1,-1,1] #not a and not b and not c
    reclauses, encodings = [], []
    for b, clause in enumerate(clauses):
      s = 's' + str(idx+b)
      varmap[varidx + idx + 1 + b] = s
      i, j, k = clause
      if i > 0 and j > 0 and k > 0: reclauses.append((i, j, k)); encodings.append(o3); subst = nary_func_to_num(lambda a, b: a | b, 2)
      elif i < 0 and j < 0 and k < 0: reclauses.append((i, j, k)); encodings.append(om3); subst = nary_func_to_num(lambda a, b, c: (1-a) & (1-b) & (1-c), 3)
      elif i > 0 and j > 0 and k < 0: reclauses.append((i, j, k)); encodings.append(o2m1); subst = nary_func_to_num(lambda a, b: a | b, 2)
      elif i > 0 and j < 0 and k < 0: reclauses.append((i, j, k)); encodings.append(o1m2); subst = nary_func_to_num(lambda a, b: a | (1-b), 2)

      elif i > 0 and j < 0 and k > 0: reclauses.append((i, k, j)); encodings.append(o2m1); subst = nary_func_to_num(lambda a, b: a | b, 2)
      elif i < 0 and j > 0 and k > 0: reclauses.append((k, j, i)); encodings.append(o2m1); subst = nary_func_to_num(lambda a, b: a | b, 2)
      elif i < 0 and j > 0 and k < 0: reclauses.append((j, i, k)); encodings.append(o1m2); subst = nary_func_to_num(lambda a, b: a | (1-b), 2)
      elif i < 0 and j < 0 and k > 0: reclauses.append((k, j, i)); encodings.append(o1m2); subst = nary_func_to_num(lambda a, b: a | (1-b), 2)
      else: assert False
      i, j, k = reclauses[-1]
      submap[s] = ((varmap[abs(i)], varmap[abs(j)], varmap[abs(k)]) if i < 0 and j < 0 and k < 0 else (varmap[abs(i)], varmap[abs(j)]), subst)
  else:
    opt3, opt2m1_1, opt2m1_2, opt2m1_3, opt1m2_1, opt1m2_2, opt1m2_3, optm3 = [all_ternary_qubo[0][nary_func_to_num(ternsat, 3)][1:len(allbinarysubs)+1] for ternsat in allternarysat]
    s = 's' + str(idx)
    varmap[varidx + idx + 1] = s
    reclauses, encodings = [], []
    for clause in clauses:
      i, j, k = clause
      i, j, k = abs(i), abs(j), abs(k)
      if (i, j) == key: i, j, k = clause #nothing
      elif (j, i) == key: j, i, k = clause #swap i,j
      elif (i, k) == key: i, k, j = clause #swap j,k
      elif (k, i) == key: j, k, i = clause #i=k,j=i
      elif (j, k) == key: k, i, j = clause #i=j,j=k
      elif (k, j) == key: k, j, i = clause #swap i,k
      else: assert False
      reclauses.append((i, j, k))
      if i > 0 and j > 0 and k > 0: encodings.append(opt3)
      elif i < 0 and j < 0 and k < 0: encodings.append(optm3)
      elif i > 0 and j > 0 and k < 0: encodings.append(opt2m1_1)
      elif i > 0 and j < 0 and k < 0: encodings.append(opt1m2_1)

      elif i > 0 and j < 0 and k > 0: encodings.append(opt2m1_2)
      elif i < 0 and j > 0 and k > 0: encodings.append(opt2m1_3)
      elif i < 0 and j > 0 and k < 0: encodings.append(opt1m2_2)
      elif i < 0 and j < 0 and k > 0: encodings.append(opt1m2_3)
      else: assert False
    #ours - a, b, c: [1-x_k+x_i*x_j-x_i*s-x_j*s+x_k*s for x_i, x_j, x_k, s in [(x_i,x_j,x_k,gen(x_i or x_j)) for gen in (lambda z: z, lambda z: 1-z) for x_i, x_j, x_k in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))]]
    #ours - ~a, ~b, ~c: [x_i*x_j-x_i*s-x_j*s+x_k*s+s for x_i, x_j, x_k, s in [(x_i,x_j,x_k,gen(x_i and x_j)) for gen in (lambda z: z, lambda z: 1-z) for x_i, x_j, x_k in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))]]
    summed = [(i, np.sum(np.vstack([encoding[i] for encoding in encodings]), axis=0)) for i in range(len(allbinarysubs)) if all(not encoding[i] is None for encoding in encodings)]
    optidx = max((x for x in summed), key=lambda x: np.count_nonzero(x[1][:-1] == 0))[0]
    encodings = [encoding[optidx] for encoding in encodings]
    submap[s] = ((varmap[key[0]], varmap[key[1]]), nary_func_to_num(lambda a, b: allbinarysubs[optidx](a, b, None), 2))
  for b, clause in enumerate(reclauses):
    s = varidx+idx+1+(b if nusslein else 0)
    i, j, k = clause    
    i, j, k = abs(i), abs(j), abs(k)
    C = encodings[b]
    c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = C
    add_bqm(bqm, (varmap[i], varmap[i]), c_0)
    add_bqm(bqm, (varmap[j], varmap[j]), c_1)
    add_bqm(bqm, (varmap[k], varmap[k]), c_2)
    add_bqm(bqm, (varmap[i], varmap[j]), c_3)
    add_bqm(bqm, (varmap[i], varmap[k]), c_4)
    add_bqm(bqm, (varmap[j], varmap[k]), c_5)
    add_bqm(bqm, (varmap[i], varmap[s]), c_6)
    add_bqm(bqm, (varmap[j], varmap[s]), c_7)
    add_bqm(bqm, (varmap[k], varmap[s]), c_8)
    add_bqm(bqm, (varmap[s], varmap[s]), c_9)
    const += c_10
  return bqm, const, idx+len(clauses) if nusslein else idx+1
def find_two_sat_in_three(cnf):
  import itertools
  two_clause = {}
  for clause in cnf:
    if len(clause) != 2: continue
    key = frozenset(abs(x) for x in clause)
    if not key in two_clause: two_clause[key] = []
    two_clause[key].append(clause)
  clause_groups = []
  for clause in cnf:
    if len(clause) <= 2: continue
    a, b, c = clause
    a, b, c = abs(a), abs(b), abs(c)
    key1, key2, key3 = frozenset((a, b)), frozenset((b, c)), frozenset((a, c))
    clause_group = [clause]
    if key1 in two_clause: clause_group.extend(two_clause[key1])
    if key2 in two_clause: clause_group.extend(two_clause[key2])
    if key3 in two_clause: clause_group.extend(two_clause[key3])
    if len(clause_group) >= 2: clause_groups.append(clause_group)
  return clause_groups
def find_three_sat_common_two(cnf, excluded):
  mp, revmap, clause_map = {}, {}, {}
  def add_map(key):
    if key[0] > key[1]: key = (key[1], key[0])
    if not key in mp: mp[key] = 0
    mp[key] += 1
  def add_clause_map(key, val):
    if key[0] > key[1]: key = (key[1], key[0])
    if not key in clause_map: clause_map[key] = []
    clause_map[key].append(val)
  def del_clause_map(key, clause, exclude):
    if key[0] > key[1]: key = (key[1], key[0])
    if key != exclude:
      clause_map[key].remove(clause)
      revmap[mp[key]].remove(key)
      if len(revmap[mp[key]]) == 0: del revmap[mp[key]]
      if mp[key]-1 != 0:
        if not mp[key]-1 in revmap: revmap[mp[key]-1] = []
        revmap[mp[key]-1].append(key)
      mp[key] -= 1
  for clause in cnf:
    if len(clause) <= 2 or clause in excluded: continue
    a, b, c = clause
    a, b, c = abs(a), abs(b), abs(c)
    add_map((a,b)); add_map((b,c)); add_map((a,c))
    add_clause_map((a,b), clause); add_clause_map((b,c), clause); add_clause_map((a,c), clause)
  for key in mp:
    if not mp[key] in revmap: revmap[mp[key]] = []
    revmap[mp[key]].append(key)
  clause_groups = []
  while True:
    if len(revmap) == 0: break
    mx = max(revmap)
    clause_group = revmap[mx].pop()
    if len(revmap[mx]) == 0: del revmap[mx]
    clause_groups.append((clause_group, clause_map[clause_group]))
    for del_clause in clause_map[clause_group]:
      a, b, c = del_clause
      a, b, c = abs(a), abs(b), abs(c)
      del_clause_map((a, b), del_clause, clause_group)
      del_clause_map((b, c), del_clause, clause_group)
      del_clause_map((a, c), del_clause, clause_group)
  return clause_groups
def random_k_sat(n, clauses, k=3):
  import numpy as np
  return ((np.random.randint(0, 2, (clauses, k))*2-1) * np.vstack([np.random.choice(np.arange(1, n+1), (k,), replace=False) for _ in range(clauses)])).tolist()
def check_random_three_sat():
  vs, subvars, subvarsold, nzeros, nzerosold, physqbits, physqbitsold, qbsolvs, qbsolvsold, dwaves, dwavesold = [], [], [], [], [], [], [], [], [], [], []
  #export DWAVE_API_TOKEN=<token>
  from dwave.system import DWaveSampler, FixedEmbeddingComposite
  from dwave.preprocessing import ScaleComposite  
  import minorminer, dimod
  adv_sampler = DWaveSampler(solver=dict(topology__type="pegasus"), token=None)
  iterations = 20
  k = 8
  nusslein_vars = lambda x: (1 if x==3 else 0) if x <= 3 else x.bit_length()+nusslein_vars(x.bit_length())
  #n+int(4.26*n)*nusslein_vars(k)=5614/8 or n(1+4.26*nusslein_vars(k))=5614/8
  experiment_sizes = [8, 16, 32, 48, 64, 80, 96, 112, 128, 136] if k == 3 else [x for x in range(k, int(5614/8/(1+4.26*nusslein_vars(k)))+1, 2)]
  qbsolvminer_sizes = experiment_sizes #list(range(3, 96+1))
  import os
  fname = "qubosat" + str(k)
  if os.path.exists(fname + ".p"):
    with open(fname + ".p", 'rb') as f:
      import pickle
      vs, subvars, subvarsold, nzeros, nzerosold, physqbits, physqbitsold, qbsolvs, qbsolvsold, dwaves, dwavesold = pickle.load(f)
  for n in range(k if len(vs)==0 else vs[-1]+1, 384+1):
    vs.append(n)
    subvar, subvarold, nzero, nzeroold, countsat, countsatold, physqbit, physqbitold, dcountsat, dcountsatold = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    print("Starting", n)
    for _ in range(iterations):
      #3-SAT: var*cvr=clauses
      while True:
        cnf = random_k_sat(n, int(4.26*n), k)
        #solved_vars, cnf = reduce_sat(cnf)
        if len(cnf) != 0 and all(len(clause)!=0 for clause in cnf): break
      #clause_groups = find_three_sat_common_two(cnf, [])
      #subvar += len(clause_groups)
      #subvarold += sum(1 for clause in cnf if len(clause)==3)
      #varmap = {i+1: str('x') + str(i) for i in range(n)}
      #bqm, const, varmap, submap = max_three_sat_to_qubo(cnf, varmap, False)
      #nzero += sum(1 for x in bqm if bqm[x] != 0)
      #varmapold = {i+1: str('x') + str(i) for i in range(n)}
      #bqmold, constold, varmapold, submapold = max_three_sat_to_qubo(cnf, varmapold, True)
      #nzeroold += sum(1 for x in bqmold if bqmold[x] != 0)
      varmap = {i+1: str('x') + str(i) for i in range(n)}
      bqm, const, varmap, submap = max_ksat_to_qubo(cnf, varmap, False)
      subvar += len(varmap)-n
      nzero += sum(1 for x in bqm if bqm[x] != 0)
      varmapold = {i+1: str('x') + str(i) for i in range(n)}
      bqmold, constold, varmapold, submapold = max_ksat_to_qubo(cnf, varmapold, True)
      subvarold += len(varmapold)-n
      nzeroold += sum(1 for x in bqmold if bqmold[x] != 0)
      
      revvarmap = {str('x') + str(i): i+1 for i in range(n)}
      if n in qbsolvminer_sizes:
        sat, sol = do_qbsolv(bqm, None) #const
        checkmaxsat = {revvarmap[x] if not x in sol or sol[x]!=0 else -revvarmap[x] for x in revvarmap}
        countsat += sum(1 for clause in cnf if any(x in checkmaxsat for x in clause))
        sat, sol = do_qbsolv(bqmold, None) #constold
        checkmaxsat = {revvarmap[x] if not x in sol or sol[x]!=0 else -revvarmap[x] for x in revvarmap}
        countsatold += sum(1 for clause in cnf if any(x in checkmaxsat for x in clause))
        print("qbsolv", n, countsat, countsatold, subvar, subvarold, nzero, nzeroold)

      if n in qbsolvminer_sizes and not adv_sampler is None:
        dwavebqm = dimod.binary.as_bqm(bqm, 'BINARY')
        embedding = minorminer.find_embedding(
          dimod.to_networkx_graph(dwavebqm), adv_sampler.to_networkx_graph())
        physqbit += len(set.union(*(set(embedding[x]) for x in embedding)))
        dwavebqmold = dimod.binary.as_bqm(bqmold, 'BINARY')
        embeddingold = minorminer.find_embedding(
          dimod.to_networkx_graph(dwavebqmold), adv_sampler.to_networkx_graph())
        physqbitold += len(set.union(*(set(embeddingold[x]) for x in embeddingold)))
        if n in experiment_sizes:
          chain_strength = max(abs(bqm[x]) for x in bqm)
          sampleset = FixedEmbeddingComposite(
                      ScaleComposite(adv_sampler),
                      embedding=embedding,
                  ).sample(
                      dwavebqm,
                      quadratic_range=adv_sampler.properties["extended_j_range"],
                      bias_range=adv_sampler.properties["h_range"],
                      chain_strength=chain_strength,
                      num_reads=100,
                      auto_scale=False,
                      label="Max-3-SAT",
                  )
          sol = sampleset.samples(1)[0]
          checkmaxsat = {revvarmap[x] if not x in sol or sol[x]!=0 else -revvarmap[x] for x in revvarmap}
          dcountsat += sum(1 for clause in cnf if any(x in checkmaxsat for x in clause))

          chain_strength = max(abs(bqmold[x]) for x in bqmold)
          sampleset = FixedEmbeddingComposite(
                      ScaleComposite(adv_sampler),
                      embedding=embeddingold,
                  ).sample(
                      dwavebqmold,
                      quadratic_range=adv_sampler.properties["extended_j_range"],
                      bias_range=adv_sampler.properties["h_range"],
                      chain_strength=chain_strength,
                      num_reads=100,
                      auto_scale=False,
                      label="Max-3-SAT",
                  )
          sol = sampleset.samples(1)[0]
          checkmaxsat = {revvarmap[x] if not x in sol or sol[x]!=0 else -revvarmap[x] for x in revvarmap}
          dcountsatold += sum(1 for clause in cnf if any(x in checkmaxsat for x in clause))
          print("dwave", n, dcountsat, dcountsatold, physqbit, physqbitold)
    subvars.append(subvar/iterations); subvarsold.append(subvarold/iterations)
    nzeros.append(nzero/iterations); nzerosold.append(nzeroold/iterations)
    if n in qbsolvminer_sizes:
      qbsolvs.append(countsat/iterations); qbsolvsold.append(countsatold/iterations)
      physqbits.append(physqbit/iterations); physqbitsold.append(physqbitold/iterations)
    if n in experiment_sizes: dwaves.append(dcountsat/iterations); dwavesold.append(dcountsatold/iterations)    
    with open(fname + ".p", 'wb') as f:
      import pickle
      pickle.dump((vs, subvars, subvarsold, nzeros, nzerosold, physqbits, physqbitsold, qbsolvs, qbsolvsold, dwaves, dwavesold), f)
def random_sat_plots():
  import os
  nusslein_vars = lambda x: (1 if x==3 else 0) if x <= 3 else x.bit_length()+nusslein_vars(x.bit_length())
  k = 3
  experiment_sizes = [8, 16, 32, 48, 64, 80, 96, 112, 128, 136] if k == 3 else [x for x in range(k, int(5614/8/(1+4.26*nusslein_vars(k)))+1, 2)]
  qbsolvminer_sizes = experiment_sizes  
  fname = "qubosat" + str(k)
  if os.path.exists(fname + ".p"):
    with open(fname + ".p", 'rb') as f:
      import pickle
      vs, subvars, subvarsold, nzeros, nzerosold, physqbits, physqbitsold, qbsolvs, qbsolvsold, dwaves, dwavesold = pickle.load(f)
  k = 8
  experiment_sizes8 = [8, 16, 32, 48, 64, 80, 96, 112, 128, 136] if k == 3 else [x for x in range(k, int(5614/8/(1+4.26*nusslein_vars(k)))+1, 2)]
  qbsolvminer_sizes8 = experiment_sizes8
  fname = "qubosat" + str(k)
  if os.path.exists(fname + ".p"):
    with open(fname + ".p", 'rb') as f:
      import pickle
      vs8, subvars8, subvarsold8, nzeros8, nzerosold8, physqbits8, physqbitsold8, qbsolvs8, qbsolvsold8, dwaves8, dwavesold8 = pickle.load(f)
  import matplotlib.pyplot as plt
  fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(7,3.6))
  #ax.set_title("Comparison of Variables in QUBO Encoding")
  ax1.set_xlabel("SAT Variables")
  ax1.set_ylabel("Substitution Variables")
  ax1.plot(vs[::16], subvarsold[::16], label="Old 3-SAT", marker="o")
  ax1.plot(vs[::16], subvars[::16], label="New 3-SAT", marker="*")
  ax1.plot(vs8[::16], subvarsold8[::16], label="Old 8-SAT", marker="+")
  ax1.plot(vs8[::16], subvars8[::16], label="New 8-SAT", marker="x")
  ax1.legend()
  #fig.savefig("qubosatvar.png", format='png')
  #fig.savefig("qubosatvar.svg", format='svg')
  #fig, ax = plt.subplots()
  #ax.set_title("Comparison of Couplings in QUBO")
  ax2.set_xlabel("SAT Variables")
  ax2.set_ylabel("Non-Zero Couplings")
  ax2.plot(vs[::16], nzerosold[::16], label="Old 3-SAT", marker="o")
  ax2.plot(vs[::16], nzeros[::16], label="New 3-SAT", marker="*")
  ax2.plot(vs8[::16], nzerosold8[::16], label="Old 8-SAT", marker="+")
  ax2.plot(vs8[::16], nzeros8[::16], label="New 8-SAT", marker="x")
  ax2.legend()
  #fig.savefig("qubosatcoupling.png", format='png')
  #fig.savefig("qubosatcoupling.svg", format='svg')
  fig.tight_layout()
  fig.savefig("qubosatqubits.png", format='png')
  fig.savefig("qubosatqubits.svg", format='svg')
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3.6))
  #ax.set_title("Comparison of Physical Qubits Required")
  ax1.set_xlabel("SAT Variables")
  ax1.set_ylabel("Physical Qubits")
  ax1.plot(qbsolvminer_sizes, physqbitsold, label="Old 3-SAT", marker="o")
  ax1.plot(qbsolvminer_sizes, physqbits, label="New 3-SAT", marker="*")
  ax1.plot(qbsolvminer_sizes8, physqbitsold8, label="Old 8-SAT", marker="+")
  ax1.plot(qbsolvminer_sizes8, physqbits8, label="New 8-SAT", marker="x")
  ax1.legend()
  ax2.set_xlabel("SAT Variables")
  ax2.set_ylabel("Satisfied Clauses")
  ax2.plot(qbsolvminer_sizes, qbsolvsold, label="Old QbSolv 3-SAT", marker="o")
  ax2.plot(qbsolvminer_sizes, qbsolvs, label="New QbSolv 3-SAT", marker="*")
  ax2.plot(experiment_sizes, dwavesold, label="Old DWave 3-SAT", marker="+")
  ax2.plot(experiment_sizes, dwaves, label="New DWave 3-SAT", marker="x")
  #ax2.plot(qbsolvminer_sizes8, qbsolvsold8, label="Old QbSolv 8-SAT", marker="1")
  #ax2.plot(qbsolvminer_sizes8, qbsolvs8, label="New QbSolv 8-SAT", marker="2")
  ax2.plot(experiment_sizes8, dwavesold8, label="Old DWave 8-SAT", marker="v")
  ax2.plot(experiment_sizes8, dwaves8, label="New DWave 8-SAT", marker="^")
  ax2.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig("qubosatexp.png", format='png')
  fig.savefig("qubosatexp.svg", format='svg')
def find_covering_clause_groups(cnf, formmaps):
  import settrie
  trie = settrie.SetTrieMap()
  for clause in cnf:
    key = set(abs(x) for x in clause)
    if not key in trie:
      trie.add((key, set()))
    trie.get(key).add(clause)
  clausegroups = []
  for formmap in formmaps:
    clausegroup = []
    for k, val in trie.itersubsets(formmap):
      trie.remove(k)
      for c in val:
        clausegroup.append(c)
    clausegroups.append(clausegroup)
  """
  clausesizes = {}
  for clause in cnf:
    if not len(clause) in clausesizes: clausesizes[len(clause)] = set()
    clausesizes[len(clause)].add(clause)
  for size in sorted(clausesizes, reverse=True):
    while len(clausesizes[size]) != 0:
      clause = next(iter(clausesizes[size]))
      key, excludekey = set(abs(x) for x in clause), set()
      excludekey |= key
      clausegroup = []
      while len(key) != 0:
        subkey = key.pop()
        allsupers = trie.itersupersets({subkey})
        supkeys = []
        for k, val in allsupers:
          if len(k.intersection(excludekey)) == 1: continue
          trie.remove(k)
          supkeys.append(k)
          for c in val:
            clausesizes[len(c)].remove(c)
            clausegroup.append(c)
        for supkey in supkeys:
          allsubsets = trie.itersubsets(supkey)
          for k, val in allsubsets:
            trie.remove(k)
            key |= k - excludekey
            excludekey |= k
            for c in val:
              clausesizes[len(c)].remove(c)
              clausegroup.append(c)
      #print(len(clausegroup), len(excludekey))
      clausegroups.append(clausegroup)
  """
  return clausegroups
def max_ksat_to_qubo(cnf, varmap, nusslein=False, formmaps=None):
  bqm, const, idx, varidx = {}, 0, 0, max(max({abs(y) for x in cnf for y in x}), max(varmap))
  submap = {}
  if nusslein:
    cnf3 = [x for x in cnf if len(x)<=3]
    s = [x for x in cnf if len(x)>3]
    while len(s) != 0:
      C = s.pop()
      h = (len(C)+1).bit_length()
      X = ['a' + str(idx+b) for b in range(h)]
      newC = [varidx + idx + 1 + b for b in range(h)]
      for b in range(h):
        varmap[varidx + idx + 1 + b] = X[b]
        submap[X[b]] = ([varmap[x if x > 0 else -x] for x in C], nary_func_to_num(lambda *args: get_nary_count_func({z for z in range(1,len(C)+1) if ((1<<b)&z)!=0})(*(y if x > 0 else 1-y for x, y in zip(C, args))), len(C)))
      idx += h
      nC = sum(1 for x in C if x < 0)
      terms = [(nC, None)] + [(1 if x > 0 else -1, varmap[x if x > 0 else -x]) for x in C] + [(-(1<<i), x) for i, x in enumerate(X)]
      for t1 in terms: #squaring
        for t2 in terms:
          if t1[1] is None and t2[1] is None: const += t1[0]*t2[0]
          elif t1[1] is None:
            add_bqm(bqm, (t2[1], t2[1]), t1[0]*t2[0])
          elif t2[1] is None:
            add_bqm(bqm, (t1[1], t1[1]), t1[0]*t2[0])
          else:
            add_bqm(bqm, (t1[1], t2[1]), t1[0]*t2[0])
      if h > 3: s.append(newC)
      else: cnf3.append(newC)
  else:
    if not formmaps is None:
      clausegroups = find_covering_clause_groups(cnf, formmaps)
      ilpdict = {}
      for clausegroup in clausegroups:
        allvars = {abs(x) for c in clausegroup for x in c}
        cglen = len(allvars)
        #print(cglen)
        vardict = {v: i for i, v in enumerate(sorted(allvars))}
        naryfunc = lambda *args: int(all(any(1-args[vardict[-x]] if x < 0 else args[vardict[x]] for x in c) for c in clausegroup))
        funckey = (nary_func_to_num(naryfunc, cglen), cglen)
        if not funckey in ilpdict:
          ilpdict[funckey] = ilp_ideal_qubo_encoding(naryfunc, cglen, [[]])[0]
          if ilpdict[funckey] is None:
            ilpdict[funckey] = ilp_ideal_qubo_encoding_allsubs(naryfunc, cglen, 1)
            assert not ilpdict[funckey] is None
            optencoding = ilp_ideal_qubo_encoding(naryfunc, cglen, [[get_nary_func(x, cglen) for x in ilpdict[funckey][1]]])[0]
            ilpdict[funckey] = (optencoding, ilpdict[funckey][1])
        encoding = ilpdict[funckey]
        if isinstance(encoding, tuple):
          encoding, subfuncs = encoding
          subs = ['a' + str(idx+b) for b in range(len(subfuncs))]
          for i in range(len(subfuncs)):
            varmap[varidx + idx + 1 + i] = subs[i]
            submap[subs[i]] = ([varmap[x if x > 0 else -x] for x in sorted(allvars)], subfuncs[i])
          idx += len(subfuncs)
          startidx = len(allvars) * (len(allvars)+1) // 2
          for s in subs:
            for x in sorted(allvars):
              add_bqm(bqm, (varmap[x if x > 0 else -x], s), encoding[startidx])
              startidx += 1              
          mixidx = startidx + len(subs)
          for i in range(len(subs)):
            add_bqm(bqm, (subs[i], subs[i]), encoding[startidx+i])
            for j in range(i+1, len(subs)):
              add_bqm(bqm, (subs[i], subs[j]), encoding[mixidx])
              mixidx += 1
        const += encoding[-1]
        mixidx = len(allvars)
        for i, x in enumerate(sorted(allvars)):
          k1 = varmap[x if x > 0 else -x]
          add_bqm(bqm, (k1, k1), encoding[i])
          for j, y in enumerate(sorted(allvars)):
            if i >= j: continue
            k2 = varmap[y if y > 0 else -y]
            add_bqm(bqm, (k1, k2), encoding[mixidx])
            mixidx += 1
        for c in clausegroup:
          cnf.remove(c)
    cnf3 = [x for x in cnf if len(x)<=3]
    s = [x for x in cnf if len(x)>3]
    for C in s:
      covsets, b, subvars, subisolates, submix = cnf_linalg_solve(len(C))
      subs = ['a' + str(idx+b) for b in range(len(subisolates))]
      for i in range(len(subisolates)):
        varmap[varidx + idx + 1 + i] = subs[i]
        submap[subs[i]] = ([varmap[x if x > 0 else -x] for x in C], nary_func_to_num(lambda *args: get_nary_count_func(covsets[i])(*(y if x > 0 else 1-y for x, y in zip(C, args))), len(C)))
      idx += len(subisolates)
      const += b[2]
      for i, x in enumerate(C):
        k1 = varmap[x if x > 0 else -x]
        add_bqm(bqm, (k1, k1), b[0])
        for j, y in enumerate(C):
          if i >= j: continue
          k2 = varmap[y if y > 0 else -y]
          add_bqm(bqm, (k1, k2), b[1])
        for j in range(len(subisolates)):
          add_bqm(bqm, (k1, subs[j]), subvars[j])
      mixidx = 0
      for i in range(len(subisolates)):
        add_bqm(bqm, (subs[i], subs[i]), subisolates[i])
        for j in range(i+1, len(subisolates)):
          add_bqm(bqm, (subs[i], subs[j]), submix[mixidx])
          mixidx += 1
      for i, x in enumerate(C): #correct negative literals
        if x > 0: continue
        k1 = varmap[-x]
        add_bqm(bqm, (k1, k1), -2*b[0]) #c(1-x) - cx = c-2cx
        const += b[0]
        for j, y in enumerate(C):
          if i == j: continue
          if y < 0: #c(1-v)(1-x)-cvx = c - cx - cv
            if i > j: continue
            k2 = varmap[-y]
            add_bqm(bqm, (k1, k1), -b[1])
            add_bqm(bqm, (k2, k2), -b[1])
            const += b[1]
          else:
            k2 = varmap[y]
            add_bqm(bqm, (k1, k2), -2*b[1]) #cv(1-x) - cvx = cv - 2cvx
            add_bqm(bqm, (k2, k2), b[1])
        for j in range(len(subisolates)):
          add_bqm(bqm, (k1, subs[j]), -2*subvars[j]) #cs(1-x) - csx = cs - 2csx
          add_bqm(bqm, (subs[j], subs[j]), subvars[j])
  if len(cnf3) != 0:
    bqm3, const3, varmap, submap3 = max_three_sat_to_qubo(cnf3, varmap, nusslein)
    for k in bqm3: add_bqm(bqm, k, bqm3[k])
    const += const3
    submap.update(submap3)
  return bqm, const, varmap, submap
def max_three_sat_to_qubo(cnf, varmap, nusslein=False):
  import itertools
  num_vars, num_clauses = len({abs(y) for x in cnf for y in x}), len(cnf)
  num_two_sat_clauses = sum(1 for clause in cnf if len(clause) == 2)
  clause_groups = find_two_sat_in_three(cnf)
  clause_groups3 = find_three_sat_common_two(cnf, clause_groups)
  encodings, best = all_ternary_qubo
  #print("Vars:", num_vars, "Clauses:", num_clauses, "2-Var Clauses:", num_two_sat_clauses,
  #      "3-Var Consumed by 2-Var", len(clause_groups), "3-Var with Common 2-Var Clause Groups:", len(clause_groups3))
  #print(sum(len(x) for _, x in clause_groups))
  bqm, const, idx, varidx = {}, 0, 0, max(max({abs(y) for x in cnf for y in x}), max(varmap))
  excluded = set()
  for clause_group in clause_groups:
    i, j, k = clause_group[0]
    i, j, k = abs(i), abs(j), abs(k)
    idx = sum([(1 if all(len(cl.intersection({i if a else -i, j if b else -j, k if c else -k})) != 0 for cl in clause_group) else 0)<<(4*a+2*b+c) for a, b, c in itertools.product((0,1), repeat=3)])
    C = encodings[idx][best[idx]]
    #assert (C[6:10] == 0).all()
    #c_0, c_1, c_2, c_3, c_4, c_5, _, _, _, _, c_10 = C
    c_0, c_1, c_2, c_3, c_4, c_5, c_10 = C
    add_bqm(bqm, (varmap[i], varmap[i]), c_0)
    add_bqm(bqm, (varmap[j], varmap[j]), c_1)
    add_bqm(bqm, (varmap[k], varmap[k]), c_2)
    add_bqm(bqm, (varmap[i], varmap[j]), c_3)
    add_bqm(bqm, (varmap[i], varmap[k]), c_4)
    add_bqm(bqm, (varmap[j], varmap[k]), c_5)
    const += c_10
    excluded.update(clause_group)
  submap = {}
  for key, clauses in clause_groups3:
    bqm, const, idx = max_three_sat_qubo(clauses, key, varidx, idx, varmap, bqm, const, submap, nusslein=nusslein)
  for clause in cnf:
    if len(clause) >= 2 or clause in excluded: continue
    if len(clause) == 1: bqm, const = max_one_sat_qubo(clause, varmap, bqm, const)
    else: bqm, const = max_two_sat_qubo(clause, varmap, bqm, const)
  return bqm, const, varmap, submap
def do_qbsolv(bqm, const):
    from dwave_qbsolv import QBSolv
    response = QBSolv().sample_qubo(bqm, target=None if const is None else -const, algorithm=None, num_repeats=1000)
    #print("samples=" + str(list(response.samples())))
    print("energies=" + str(list(response.data_vectors['energy'])))
    energy = response.data_vectors['energy'][0]
    smpl = dict(response.samples()[0])    
    if const is None: return None, smpl
    print(energy + const, smpl)
    sat = energy + const == 0
    return sat, smpl
#check_random_three_sat(); assert False
#random_sat_plots(); assert False
