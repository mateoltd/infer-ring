//

import Foundation
import Ring

struct Ping: Codable {
    let isAlive: Bool
}

struct ElectionMessage: Codable {
    let type: ElectionMessageType
    let candidateID: DeviceID
    let hardwareProfile: HardwareProfile
    let timestamp: Date
}

enum ElectionMessageType: Codable {
    case election
    case coordinator
}

struct ModelLoadRequest: Codable {
    let modelCard: ModelCard
    let availableFiles: [String]
    let shardMeta: ShardMetadata
    let enableQwenMTP: Bool
    let requestID: String
    let timestamp: Date
}

struct ModelLoadResponse: Codable {
    let requestID: String
    let success: Bool
    let errorMessage: String?
    let timestamp: Date
}

struct DebugModelLoadRequest: Codable {
    let modelId: String
    let requestID: String
}

struct DebugRingDevice: Codable {
    let name: String
    let host: String
    let rank: Int?
    let recommendedRAM: Int?
}

struct DebugRingSnapshot: Codable {
    let state: String
    let isLeader: Bool
    let usableRAM: Int
    let discovered: [DebugRingDevice]
    let ring: [DebugRingDevice]
}

struct GenerationRequest: Codable {
    let requestID: String
    let input: String
    let inputRole: ChatMessage.Role
    let history: [OpenAPIMessage]?
    let tools: [OpenAPITool]?
    let maxTokens: Int?
    let timestamp: Date
}

struct GenerationResponse: Codable {
    let requestID: String
    let success: Bool
    let errorMessage: String?
    let timestamp: Date
}

struct ChatResetRequest: Codable {
    let requestID: String
    let timestamp: Date
}

struct ChatResetResponse: Codable {
    let requestID: String
    let success: Bool
    let errorMessage: String?
    let timestamp: Date
}

struct HardwareProfileRequest: Codable {
    let timestamp: Date
}

struct HardwareProfileResponse: Codable {
    let hardwareProfile: HardwareProfile
    let timestamp: Date
}
