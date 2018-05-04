.section .text

# In Linux' calling convention, the parameters are in rdi, rsi, rdx, rcx, r8, r9.

# Signature:
#     uint64 Mul64(uint64 a, uint64 b, uint64 * phi)
# Returns the low bits and stuffs the high bits in *phi.
.globl Mul64
.type Mul64, @function
Mul64:
    mov %rdx,%r9 # rdx will be used in mul
    mov %rdi,%rax
    mul %rsi
    mov %rdx,(%r9)
    retq

# Signature:
#     uint64 Div64(uint64 lo, uint64 hi, uint64 den, uint64 * prem)
# Returns the quotient and stuffs the remainder in *prem.
.globl Div64
.type Div64, @function
Div64:
    mov %rdx,%r9 # rdx will be used in div
    mov %rdi,%rax
    mov %rsi,%rdx
    div %r9
    mov %rdx,(%rcx)
    retq

# Signature:
#     uint64 MulDiv64Core(uint64 a, uint64 b, uint64 den, uint64 * prem)
# Returns the quotient and stuffs the remainder in *prem.
.globl MulDiv64Core
.type MulDiv64Core, @function
MulDiv64Core:
    mov %rdx,%r9 # rdx will be used in mul.
    mov %rdi,%rax
    mul %rsi
    div %r9
    mov %rdx,(%rcx)
    retq

# Signature:
#     bool TryMulDiv64Core(uint64 a, uint64 b, uint64 den, uint64 * pquo, uint64 * prem)
# Stuffs the quotient in *pquo and the remainder in *prem. Returns false if the quotient
# overflows or den is zero.
.globl TryMulDiv64Core
.type TryMulDiv64Core, @function
TryMulDiv64Core:
    # Do the multiplication.
    mov %rdx,%r9 # rdx will be used in mul.
    mov %rdi,%rax
    mul %rsi
    
    # Test whether rdx is bigger than or equal to den. Note that this
    # handles the case when den is zero.
    cmp %r9,%rdx
    jae LOverflow

    # Safe to do the division.
    div %r9

    # Store the results and return true.
    mov %rax,(%rcx)
    mov %rdx,(%r8)
    movb $0x1,%al
    retq

LOverflow:
    # The denominator is zero, so set *pquo and *prem to zero and return false.
    xor %rax,%rax
    mov %rax,(%rcx)
    mov %rax,(%r8)
    retq