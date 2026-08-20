// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "./CarbonCreditToken.sol";

/**
 * @title  BlueCarbonRegistry
 * @notice On-chain registry for blue-carbon MRV sites (mangroves, seagrass, marshes).
 *
 *         Workflow:
 *           1. NGO / community calls registerProject() to register a coastal site.
 *           2. Oracle / owner calls verifyAndIssueCredits() after the off-chain
 *              ML pipeline (Python) produces a carbon estimate and IPFS proof hash.
 *           3. BCO2 tokens are minted directly to the project owner's wallet.
 *           4. Any token holder calls retireCredits() to permanently burn credits
 *              (e.g. corporate ESG offset), creating an immutable retirement record.
 *
 * @dev    Owner of this contract acts as the trusted verification oracle.
 *         Integration with Smart India Hackathon PS-25038 —
 *         Ministry of Earth Sciences Blue Carbon MRV prototype.
 */
contract BlueCarbonRegistry is Ownable {

    // ── Data structures ──────────────────────────────────────────────────────
    struct Project {
        string  siteId;           // Unique identifier (e.g. "BHIT-001")
        address owner;            // NGO / community wallet
        string  latitude;         // Decimal degrees (stored as string for precision)
        string  longitude;
        uint256 areaHectares;     // Mangrove area in hectares
        bool    isVerified;       // True once oracle has verified
        uint256 carbonTons;       // Predicted carbon stock (metric tons CO2e)
        uint256 creditsMinted;    // BCO2 tokens minted (18 decimals)
        string  ipfsProofHash;    // IPFS CID / URL for ML output + Sentinel-2 metadata
        uint256 timestamp;        // Block timestamp of registration
    }

    // ── State ────────────────────────────────────────────────────────────────
    /// @notice Lookup a project by its siteId string key.
    mapping(string => Project) public projects;

    /// @notice Ordered list of all registered siteIds (for enumeration).
    string[] public allSiteIds;

    /// @notice Address of the BCO2 ERC-20 token contract.
    address public tokenContractAddress;

    /// @dev    Typed reference to the token contract for direct calls.
    CarbonCreditToken private carbonToken;

    // ── Events ───────────────────────────────────────────────────────────────
    event ProjectRegistered(
        string  indexed siteId,
        address indexed owner,
        string  latitude,
        string  longitude,
        uint256 areaHectares
    );

    event ProjectVerified(
        string  indexed siteId,
        uint256 carbonTons,
        string  ipfsProofHash
    );

    event CreditsIssued(
        string  indexed siteId,
        address indexed owner,
        uint256 amount          // in token wei (18 decimals)
    );

    event CreditsRetired(
        address indexed burner,
        uint256 amount,         // in token wei (18 decimals)
        string  reason
    );

    // ── Constructor ──────────────────────────────────────────────────────────
    /**
     * @param _tokenAddress  Address of the already-deployed CarbonCreditToken.
     * @param initialOwner   Wallet that acts as the verification oracle.
     */
    constructor(address _tokenAddress, address initialOwner)
        Ownable(initialOwner)
    {
        require(_tokenAddress != address(0), "BlueCarbonRegistry: zero token address");
        tokenContractAddress = _tokenAddress;
        carbonToken          = CarbonCreditToken(_tokenAddress);
    }

    // ── Core Functions ───────────────────────────────────────────────────────

    /**
     * @notice Register a new coastal carbon site.
     * @dev    Anyone (NGO, local community) may call this.
     *         Each siteId must be globally unique.
     *
     * @param siteId        Human-readable unique identifier (e.g. "BHIT-001").
     * @param lat           Decimal latitude string (e.g. "20.7211").
     * @param lon           Decimal longitude string (e.g. "86.8880").
     * @param areaHectares  Mapped mangrove area in hectares.
     */
    function registerProject(
        string memory siteId,
        string memory lat,
        string memory lon,
        uint256 areaHectares
    ) external {
        require(bytes(siteId).length > 0,               "BlueCarbonRegistry: empty siteId");
        require(areaHectares > 0,                        "BlueCarbonRegistry: area must be > 0");
        require(
            bytes(projects[siteId].siteId).length == 0,
            "BlueCarbonRegistry: siteId already registered"
        );

        projects[siteId] = Project({
            siteId:        siteId,
            owner:         msg.sender,
            latitude:      lat,
            longitude:     lon,
            areaHectares:  areaHectares,
            isVerified:    false,
            carbonTons:    0,
            creditsMinted: 0,
            ipfsProofHash: "",
            timestamp:     block.timestamp
        });

        allSiteIds.push(siteId);

        emit ProjectRegistered(siteId, msg.sender, lat, lon, areaHectares);
    }

    /**
     * @notice Verify a site and issue BCO2 carbon credits to the project owner.
     * @dev    Only callable by the contract owner (trusted oracle).
     *         Triggers CarbonCreditToken.mint() with the predicted carbon volume.
     *         Double-verification is blocked.
     *
     * @param siteId               The registered site identifier.
     * @param predictedCarbonTons  Carbon stock estimate in metric tons CO2e.
     * @param ipfsProofHash        IPFS CID pointing to the ML verification output
     *                             and Sentinel-2 satellite metadata.
     */
    function verifyAndIssueCredits(
        string memory siteId,
        uint256 predictedCarbonTons,
        string memory ipfsProofHash
    ) external onlyOwner {
        require(
            bytes(projects[siteId].siteId).length > 0,
            "BlueCarbonRegistry: project not registered"
        );
        require(
            !projects[siteId].isVerified,
            "BlueCarbonRegistry: project already verified"
        );
        require(predictedCarbonTons > 0, "BlueCarbonRegistry: carbon tons must be > 0");

        // Update project state
        projects[siteId].isVerified    = true;
        projects[siteId].carbonTons    = predictedCarbonTons;
        projects[siteId].ipfsProofHash = ipfsProofHash;

        // Convert metric tons → token units (18 decimals, 1 token = 1 tCO2e)
        uint256 amountToMint = predictedCarbonTons * (10 ** 18);
        projects[siteId].creditsMinted = amountToMint;

        // Mint BCO2 tokens to the project owner
        carbonToken.mint(projects[siteId].owner, amountToMint);

        emit ProjectVerified(siteId, predictedCarbonTons, ipfsProofHash);
        emit CreditsIssued(siteId, projects[siteId].owner, amountToMint);
    }

    /**
     * @notice Permanently retire (burn) BCO2 carbon credits.
     * @dev    Caller must have pre-approved this contract via
     *         `CarbonCreditToken.approve(registryAddress, amount)`.
     *         Emits an immutable CreditsRetired record (e.g. for ESG reporting).
     *
     * @param amount  Number of BCO2 tokens to retire (18 decimals).
     * @param reason  Human-readable retirement reason (e.g. "Corporate ESG Offset Q3").
     */
    function retireCredits(uint256 amount, string memory reason) external {
        require(amount > 0, "BlueCarbonRegistry: amount must be > 0");
        require(bytes(reason).length > 0, "BlueCarbonRegistry: reason required");

        // Transfers tokens from caller to zero address (burn), requires prior approval
        carbonToken.burnFrom(msg.sender, amount);

        emit CreditsRetired(msg.sender, amount, reason);
    }

    // ── View Functions ───────────────────────────────────────────────────────

    /**
     * @notice Return the full project data for a given siteId.
     * @param  siteId  The unique site identifier.
     * @return project The complete Project struct.
     */
    function getProject(string memory siteId)
        external
        view
        returns (Project memory project)
    {
        require(
            bytes(projects[siteId].siteId).length > 0,
            "BlueCarbonRegistry: project not found"
        );
        return projects[siteId];
    }

    /**
     * @notice Return all registered siteIds in insertion order.
     * @return Array of siteId strings.
     */
    function getAllSiteIds() external view returns (string[] memory) {
        return allSiteIds;
    }
}
