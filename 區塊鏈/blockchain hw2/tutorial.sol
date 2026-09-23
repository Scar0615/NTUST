// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

contract MyFirstContract{
    string public s = "Hello World !!";
    uint256 public Value;

    function get() public view returns(uint){
        return Value;
    }

    function set(uint x) public {
        Value = x;
    }
}

contract SafeMath {
    function safeAdd (uint a, uint b) public pure returns (uint c) {
        c = a + b;
        require(c >= a);
    }

    function safeSub(uint a, uint b) public pure returns (uint c) {
        require(b <= a, "Invalid operation!");
        c = a - b;
    }

    function safeMul(uint a, uint b) public pure returns (uint c) {
        c = a * b;
        require(a==0 || c/a==b);
    }

    function safeDiv(uint a, uint b) public pure returns (uint c) {
        require(b > 0);
        c = a / b;
    }
}