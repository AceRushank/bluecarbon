// =============================================================================
// BlueCarbonRegistry.test.js
// =============================================================================
// Automated test suite for the Blue Carbon MRV smart contract system.
//
// Stack:  Hardhat + Ethers.js v6 + OpenZeppelin Contracts v5
// Run:    npx hardhat test
//
// Test coverage:
//   Test 1 — Project registration + metadata verification
//   Test 2 — Only owner (oracle) can verify and mint
//   Test 3 — Correct BCO2 token balance credited on verification
//   Test 4 — Credit retirement burns tokens and emits CreditsRetired
//   Test 5 — Double-verification / double-mint protection
//   Test 6 — Duplicate siteId registration reverts
//   Test 7 — getAllSiteIds returns all registered IDs
//   Test 8 — retireCredits reverts without prior token approval
// =============================================================================

const { expect }  = require("chai");
const { ethers }  = require("hardhat");

// ── Fixtures ──────────────────────────────────────────────────────────────────
// Deploy fresh contracts before each test for isolation.
async function deployFixture() {
  const [owner, ngo, corporate, other] = await ethers.getSigners();

  // 1. Deploy CarbonCreditToken
  const TokenFactory = await ethers.getContractFactory("CarbonCreditToken");
  const token = await TokenFactory.deploy(owner.address);
  await token.waitForDeployment();

  // 2. Deploy BlueCarbonRegistry (pass token address + owner)
  const RegistryFactory = await ethers.getContractFactory("BlueCarbonRegistry");
  const registry = await RegistryFactory.deploy(token.target, owner.address);
  await registry.waitForDeployment();

  // 3. Authorise registry as the sole minter on the token
  await token.connect(owner).setMinter(registry.target);

  return { token, registry, owner, ngo, corporate, other };
}

// ── Sample site data ──────────────────────────────────────────────────────────
const SITE = {
  id:          "BHIT-001",
  lat:         "20.7211",
  lon:         "86.8880",
  area:        100n,                 // 100 hectares
  carbonTons:  500n,                 // 500 tCO2e
  ipfs:        "ipfs://QmBlueCarbonProofHash001",
};

// ─────────────────────────────────────────────────────────────────────────────
describe("BlueCarbonRegistry — Full Test Suite", function () {

  // ══════════════════════════════════════════════════════════════════════════
  // Test 1: Project registration
  // ══════════════════════════════════════════════════════════════════════════
  describe("Test 1: Project Registration", function () {
    it("emits ProjectRegistered with correct args", async function () {
      const { registry, ngo } = await deployFixture();

      await expect(
        registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area)
      )
        .to.emit(registry, "ProjectRegistered")
        .withArgs(SITE.id, ngo.address, SITE.lat, SITE.lon, SITE.area);
    });

    it("stores correct metadata on-chain", async function () {
      const { registry, ngo } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);

      const p = await registry.getProject(SITE.id);
      expect(p.siteId).to.equal(SITE.id);
      expect(p.owner).to.equal(ngo.address);
      expect(p.latitude).to.equal(SITE.lat);
      expect(p.longitude).to.equal(SITE.lon);
      expect(p.areaHectares).to.equal(SITE.area);
      expect(p.isVerified).to.equal(false);
      expect(p.carbonTons).to.equal(0n);
      expect(p.creditsMinted).to.equal(0n);
    });

    it("appends siteId to allSiteIds", async function () {
      const { registry, ngo } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);

      const ids = await registry.getAllSiteIds();
      expect(ids).to.deep.equal([SITE.id]);
    });
  });

  // ══════════════════════════════════════════════════════════════════════════
  // Test 2: Only owner (oracle) can verify
  // ══════════════════════════════════════════════════════════════════════════
  describe("Test 2: Oracle-Only Verification", function () {
    it("reverts when a non-owner tries to verify a project", async function () {
      const { registry, ngo, other } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);

      // OZ v5 Ownable uses a custom error
      await expect(
        registry.connect(other).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs)
      ).to.be.revertedWithCustomError(registry, "OwnableUnauthorizedAccount");
    });

    it("owner (oracle) can verify successfully", async function () {
      const { registry, ngo, owner } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);

      await expect(
        registry.connect(owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs)
      )
        .to.emit(registry, "ProjectVerified")
        .withArgs(SITE.id, SITE.carbonTons, SITE.ipfs);
    });
  });

  // ══════════════════════════════════════════════════════════════════════════
  // Test 3: Correct BCO2 token balance credited on verification
  // ══════════════════════════════════════════════════════════════════════════
  describe("Test 3: Token Balance After Verification", function () {
    it("mints exactly predictedCarbonTons * 1e18 BCO2 to the project owner", async function () {
      const { token, registry, ngo, owner } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);
      await registry.connect(owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs);

      const expected = ethers.parseEther(SITE.carbonTons.toString());   // 500 * 1e18
      const balance  = await token.balanceOf(ngo.address);
      expect(balance).to.equal(expected);
    });

    it("emits CreditsIssued with the correct amount", async function () {
      const { registry, ngo, owner } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);

      const expectedAmount = ethers.parseEther(SITE.carbonTons.toString());

      await expect(
        registry.connect(owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs)
      )
        .to.emit(registry, "CreditsIssued")
        .withArgs(SITE.id, ngo.address, expectedAmount);
    });

    it("updates creditsMinted in the project struct", async function () {
      const { registry, ngo, owner } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);
      await registry.connect(owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs);

      const p = await registry.getProject(SITE.id);
      expect(p.creditsMinted).to.equal(ethers.parseEther(SITE.carbonTons.toString()));
      expect(p.isVerified).to.equal(true);
      expect(p.carbonTons).to.equal(SITE.carbonTons);
      expect(p.ipfsProofHash).to.equal(SITE.ipfs);
    });
  });

  // ══════════════════════════════════════════════════════════════════════════
  // Test 4: Credit retirement
  // ══════════════════════════════════════════════════════════════════════════
  describe("Test 4: Credit Retirement", function () {
    const RETIRE_AMOUNT = ethers.parseEther("100");   // retire 100 BCO2
    const RETIRE_REASON = "Corporate ESG Offset Q3 2026";

    async function mintedFixture() {
      const f = await deployFixture();
      await f.registry.connect(f.ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);
      await f.registry.connect(f.owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs);
      return f;
    }

    it("burns tokens and emits CreditsRetired with correct args", async function () {
      const { token, registry, ngo } = await mintedFixture();

      // Step 1: ngo approves the registry to spend its BCO2
      await token.connect(ngo).approve(registry.target, RETIRE_AMOUNT);

      // Step 2: retire — should emit CreditsRetired and burn tokens
      await expect(registry.connect(ngo).retireCredits(RETIRE_AMOUNT, RETIRE_REASON))
        .to.emit(registry, "CreditsRetired")
        .withArgs(ngo.address, RETIRE_AMOUNT, RETIRE_REASON);
    });

    it("reduces the caller's token balance by the retired amount", async function () {
      const { token, registry, ngo } = await mintedFixture();

      const balanceBefore = await token.balanceOf(ngo.address);
      await token.connect(ngo).approve(registry.target, RETIRE_AMOUNT);
      await registry.connect(ngo).retireCredits(RETIRE_AMOUNT, RETIRE_REASON);
      const balanceAfter = await token.balanceOf(ngo.address);

      expect(balanceAfter).to.equal(balanceBefore - RETIRE_AMOUNT);
    });

    it("reduces total BCO2 supply (tokens are burned, not transferred)", async function () {
      const { token, registry, ngo } = await mintedFixture();

      const supplyBefore = await token.totalSupply();
      await token.connect(ngo).approve(registry.target, RETIRE_AMOUNT);
      await registry.connect(ngo).retireCredits(RETIRE_AMOUNT, RETIRE_REASON);
      const supplyAfter = await token.totalSupply();

      expect(supplyAfter).to.equal(supplyBefore - RETIRE_AMOUNT);
    });
  });

  // ══════════════════════════════════════════════════════════════════════════
  // Test 5: Double-verification / double-mint protection
  // ══════════════════════════════════════════════════════════════════════════
  describe("Test 5: Double-Mint Protection", function () {
    it("reverts on second verifyAndIssueCredits call for the same project", async function () {
      const { registry, ngo, owner } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);
      await registry.connect(owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs);

      // Second attempt must revert
      await expect(
        registry.connect(owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs)
      ).to.be.revertedWith("BlueCarbonRegistry: project already verified");
    });

    it("token supply stays constant after double-mint attempt", async function () {
      const { token, registry, ngo, owner } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);
      await registry.connect(owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs);

      const supplyAfterFirst = await token.totalSupply();

      try {
        await registry.connect(owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs);
      } catch {}

      expect(await token.totalSupply()).to.equal(supplyAfterFirst);
    });
  });

  // ══════════════════════════════════════════════════════════════════════════
  // Test 6: Duplicate siteId registration reverts
  // ══════════════════════════════════════════════════════════════════════════
  describe("Test 6: Duplicate siteId Protection", function () {
    it("reverts if the same siteId is registered twice", async function () {
      const { registry, ngo, other } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);

      await expect(
        registry.connect(other).registerProject(SITE.id, "0.0", "0.0", 50n)
      ).to.be.revertedWith("BlueCarbonRegistry: siteId already registered");
    });
  });

  // ══════════════════════════════════════════════════════════════════════════
  // Test 7: getAllSiteIds enumeration
  // ══════════════════════════════════════════════════════════════════════════
  describe("Test 7: Site Enumeration", function () {
    it("returns all registered siteIds in order", async function () {
      const { registry, ngo, other } = await deployFixture();

      await registry.connect(ngo).registerProject("SUND-001",  "21.9", "89.1", 200n);
      await registry.connect(ngo).registerProject("PICH-001",  "11.4", "79.8", 50n);
      await registry.connect(other).registerProject("KUCH-001", "23.1", "68.5", 75n);

      const ids = await registry.getAllSiteIds();
      expect(ids).to.deep.equal(["SUND-001", "PICH-001", "KUCH-001"]);
      expect(ids.length).to.equal(3);
    });
  });

  // ══════════════════════════════════════════════════════════════════════════
  // Test 8: retireCredits reverts without prior approval
  // ══════════════════════════════════════════════════════════════════════════
  describe("Test 8: retireCredits Requires Prior Approval", function () {
    it("reverts with ERC20InsufficientAllowance if user has not approved registry", async function () {
      const { registry, ngo, owner } = await deployFixture();

      await registry.connect(ngo).registerProject(SITE.id, SITE.lat, SITE.lon, SITE.area);
      await registry.connect(owner).verifyAndIssueCredits(SITE.id, SITE.carbonTons, SITE.ipfs);

      // No approval → ERC-20 burnFrom will revert with OZ v5 custom error
      await expect(
        registry.connect(ngo).retireCredits(ethers.parseEther("100"), "Test reason")
      ).to.be.revertedWithCustomError(
        await ethers.getContractAt("CarbonCreditToken", await registry.tokenContractAddress()),
        "ERC20InsufficientAllowance"
      );
    });
  });

});
