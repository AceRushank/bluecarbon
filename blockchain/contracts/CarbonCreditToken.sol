// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title CarbonCreditToken ($BCO2)
 * @notice ERC-20 token representing verified blue carbon credits.
 *         1 BCO2 token = 1 metric ton of CO2-equivalent sequestered.
 *
 * @dev    Minting is exclusively controlled by a designated minter address
 *         (intended to be the BlueCarbonRegistry contract), set by the owner
 *         after deployment.  Token holders may freely burn their own credits
 *         (retirement) via the inherited ERC20Burnable functions.
 */
contract CarbonCreditToken is ERC20, ERC20Burnable, Ownable {

    /// @notice Address authorised to call mint() — must be BlueCarbonRegistry.
    address public minter;

    // ── Events ──────────────────────────────────────────────────────────────
    event MinterUpdated(address indexed previousMinter, address indexed newMinter);

    // ── Constructor ──────────────────────────────────────────────────────────
    /**
     * @param initialOwner  The deployer / admin wallet that can later call setMinter().
     */
    constructor(address initialOwner)
        ERC20("Blue Carbon Credit", "BCO2")
        Ownable(initialOwner)
    {}

    // ── Admin ────────────────────────────────────────────────────────────────
    /**
     * @notice Designates which address may call mint().
     *         Call this once after deploying BlueCarbonRegistry.
     * @param  _minter  Address of the BlueCarbonRegistry contract.
     */
    function setMinter(address _minter) external onlyOwner {
        require(_minter != address(0), "CarbonCreditToken: zero address");
        address prev = minter;
        minter = _minter;
        emit MinterUpdated(prev, _minter);
    }

    // ── Minting (registry-only) ──────────────────────────────────────────────
    /**
     * @notice Mints verified carbon credits to a project owner.
     * @dev    Only callable by the designated BlueCarbonRegistry contract.
     * @param  to      Recipient (NGO / community wallet).
     * @param  amount  Token amount in wei-equivalent (18 decimals).
     */
    function mint(address to, uint256 amount) external {
        require(msg.sender == minter, "CarbonCreditToken: caller is not the minter");
        _mint(to, amount);
    }
}
