require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },

  networks: {
    // ── Local development network (default) ──────────────────────────────────
    hardhat: {
      chainId: 31337,
    },

    // ── Ethereum Sepolia Testnet ─────────────────────────────────────────────
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "https://rpc.sepolia.org",
      accounts: process.env.PRIVATE_KEY ? [`0x${process.env.PRIVATE_KEY}`] : [],
      chainId: 11155111,
      gasPrice: "auto",
    },

    // ── Polygon Amoy Testnet ──────────────────────────────────────────────────
    // Required env vars: PRIVATE_KEY, AMOY_RPC_URL (see .env.example)
    amoy: {
      url: process.env.AMOY_RPC_URL || "https://rpc-amoy.polygon.technology/",
      accounts: process.env.PRIVATE_KEY ? [`0x${process.env.PRIVATE_KEY}`] : [],
      chainId: 80002,
      gasPrice: "auto",
    },
  },

  // ── Etherscan / Polygonscan verification ─────────────────────────────────
  etherscan: {
    apiKey: {
      sepolia: process.env.ETHERSCAN_API_KEY || "",
      polygonAmoy: process.env.POLYGONSCAN_API_KEY || "",
    },
    customChains: [
      {
        network: "polygonAmoy",
        chainId: 80002,
        urls: {
          apiURL: "https://api-amoy.polygonscan.com/api",
          browserURL: "https://amoy.polygonscan.com",
        },
      },
    ],
  },

  // ── Gas reporter (optional, set REPORT_GAS=true) ──────────────────────────
  gasReporter: {
    enabled: process.env.REPORT_GAS === "true",
    currency: "USD",
  },

  // ── Test file pattern ─────────────────────────────────────────────────────
  mocha: {
    timeout: 60000,
  },
};
