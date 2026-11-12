"""Route immutable anchor simulations and candidate-core simulations."""
import simulations as anchors
import hard_simulations as harder

def implementation(cid):
    return harder if int(cid[1:]) >= 219 else anchors

def initial(cid):return implementation(cid).initial(cid)
def view(case,n,state):return implementation(case['id']).view(case,n,state)
def commit(cid,state,answers):return implementation(cid).commit(cid,state,answers)
def advance(cid,state):return implementation(cid).advance(cid,state)
def terminal_ok(cid,state):return implementation(cid).terminal_ok(cid,state)
