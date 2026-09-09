import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { CheckCircle2, RefreshCcw } from "lucide-react";

import { useSettings } from "@/hooks/useSettings";
import { SettingContainer, SettingsGroup, ToggleSwitch } from "@/components/ui";
import { Alert } from "@/components/ui/Alert";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { ResetButton } from "@/components/ui/ResetButton";
import { Dropdown } from "@/components/ui/Dropdown";
import { ApiKeyField } from "../PostProcessingSettingsApi/ApiKeyField";
import { BaseUrlField } from "../PostProcessingSettingsApi/BaseUrlField";
import { ModelSelect } from "../PostProcessingSettingsApi/ModelSelect";
import type { ModelOption } from "../PostProcessingSettingsApi/types";

/// Settings for sending recordings to a remote speech-to-text endpoint instead
/// of the bundled local engine. Every provider here speaks the same
/// OpenAI-compatible `/audio/transcriptions` dialect, which is why hosted APIs
/// (OpenRouter, OpenAI, Groq) and self-hosted servers share one form.
const RemoteTranscriptionSettingsComponent: React.FC = () => {
  const { t } = useTranslation();
  const {
    settings,
    getSetting,
    updateSetting,
    isUpdating,
    remoteTranscriptionModelOptions,
    setRemoteTranscriptionProvider,
    updateRemoteTranscriptionBaseUrl,
    updateRemoteTranscriptionApiKey,
    updateRemoteTranscriptionModel,
    fetchRemoteTranscriptionModels,
    testRemoteTranscription,
  } = useSettings();

  const [testState, setTestState] = useState<
    { status: "idle" } | { status: "ok" } | { status: "error"; message: string }
  >({ status: "idle" });

  const enabled = getSetting("remote_transcription_enabled") ?? false;
  const providers = settings?.remote_transcription_providers ?? [];
  const providerId =
    getSetting("remote_transcription_provider_id") ??
    providers[0]?.id ??
    "openrouter";
  const provider = providers.find((p) => p.id === providerId);

  const apiKey = settings?.remote_transcription_api_keys?.[providerId] ?? "";
  // An unset model falls back to the provider's suggestion, mirroring what the
  // backend will actually send.
  const model =
    settings?.remote_transcription_models?.[providerId] ||
    provider?.default_model ||
    "";

  const providerOptions = useMemo(
    () => providers.map((p) => ({ value: p.id, label: p.label })),
    [providers],
  );

  const modelOptions = useMemo<ModelOption[]>(() => {
    const seen = new Set<string>();
    const options: ModelOption[] = [];
    const upsert = (value?: string | null) => {
      const trimmed = value?.trim();
      if (!trimmed || seen.has(trimmed)) return;
      seen.add(trimmed);
      options.push({ value: trimmed, label: trimmed });
    };
    for (const candidate of remoteTranscriptionModelOptions[providerId] ?? []) {
      upsert(candidate);
    }
    upsert(model);
    return options;
  }, [remoteTranscriptionModelOptions, providerId, model]);

  const handleTest = async () => {
    setTestState({ status: "idle" });
    try {
      await testRemoteTranscription(providerId);
      setTestState({ status: "ok" });
    } catch (error) {
      setTestState({
        status: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const isFetchingModels = isUpdating(
    `remote_transcription_models_fetch:${providerId}`,
  );
  const isTesting = isUpdating(`remote_transcription_test:${providerId}`);

  return (
    <SettingsGroup title={t("settings.remoteTranscription.title")}>
      <ToggleSwitch
        checked={enabled}
        onChange={(value) => {
          setTestState({ status: "idle" });
          void updateSetting("remote_transcription_enabled", value);
        }}
        isUpdating={isUpdating("remote_transcription_enabled")}
        label={t("settings.remoteTranscription.enable.label")}
        description={t("settings.remoteTranscription.enable.description")}
        descriptionMode="tooltip"
        grouped={true}
      />

      {enabled && (
        <>
          <SettingContainer
            title={t("settings.remoteTranscription.provider.title")}
            description={t("settings.remoteTranscription.provider.description")}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <Dropdown
              selectedValue={providerId}
              options={providerOptions}
              onSelect={(value) => {
                if (!value || value === providerId) return;
                setTestState({ status: "idle" });
                void setRemoteTranscriptionProvider(value);
              }}
              disabled={isUpdating("remote_transcription_provider_id")}
              className="min-w-[220px]"
            />
          </SettingContainer>

          {provider?.allow_base_url_edit && (
            <SettingContainer
              title={t("settings.remoteTranscription.baseUrl.title")}
              description={t(
                "settings.remoteTranscription.baseUrl.description",
              )}
              descriptionMode="tooltip"
              layout="horizontal"
              grouped={true}
            >
              <BaseUrlField
                value={provider.base_url}
                onBlur={(value) => {
                  const trimmed = value.trim();
                  if (trimmed && trimmed !== provider.base_url) {
                    void updateRemoteTranscriptionBaseUrl(providerId, trimmed);
                  }
                }}
                placeholder="http://localhost:8000/v1"
                disabled={isUpdating(
                  `remote_transcription_base_url:${providerId}`,
                )}
                className="min-w-[380px]"
              />
            </SettingContainer>
          )}

          <SettingContainer
            title={t("settings.remoteTranscription.apiKey.title")}
            description={t("settings.remoteTranscription.apiKey.description")}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <ApiKeyField
              value={apiKey}
              onBlur={(value) => {
                const trimmed = value.trim();
                if (trimmed !== apiKey) {
                  void updateRemoteTranscriptionApiKey(providerId, trimmed);
                }
              }}
              placeholder={
                provider?.requires_api_key
                  ? t("settings.remoteTranscription.apiKey.placeholder")
                  : t("settings.remoteTranscription.apiKey.placeholderOptional")
              }
              disabled={isUpdating(
                `remote_transcription_api_key:${providerId}`,
              )}
              className="min-w-[320px]"
            />
          </SettingContainer>

          <SettingContainer
            title={t("settings.remoteTranscription.model.title")}
            description={t("settings.remoteTranscription.model.description")}
            descriptionMode="tooltip"
            layout="stacked"
            grouped={true}
          >
            <div className="flex items-center gap-2">
              <ModelSelect
                value={model}
                options={modelOptions}
                disabled={isUpdating(
                  `remote_transcription_model:${providerId}`,
                )}
                isLoading={isFetchingModels}
                placeholder={t(
                  "settings.remoteTranscription.model.placeholder",
                )}
                onSelect={(value) =>
                  void updateRemoteTranscriptionModel(providerId, value.trim())
                }
                onCreate={(value) =>
                  void updateRemoteTranscriptionModel(providerId, value.trim())
                }
                onBlur={() => {}}
                className="flex-1 min-w-[380px]"
              />
              <ResetButton
                onClick={() => void fetchRemoteTranscriptionModels(providerId)}
                disabled={isFetchingModels}
                ariaLabel={t("settings.remoteTranscription.model.refresh")}
                className="flex h-10 w-10 items-center justify-center"
              >
                <RefreshCcw
                  className={`h-4 w-4 ${isFetchingModels ? "animate-spin" : ""}`}
                />
              </ResetButton>
            </div>
          </SettingContainer>

          <SettingContainer
            title={t("settings.remoteTranscription.prompt.title")}
            description={t("settings.remoteTranscription.prompt.description")}
            descriptionMode="tooltip"
            layout="stacked"
            grouped={true}
          >
            <Input
              type="text"
              defaultValue={getSetting("remote_transcription_prompt") ?? ""}
              onBlur={(event) =>
                void updateSetting(
                  "remote_transcription_prompt",
                  event.target.value,
                )
              }
              placeholder={t("settings.remoteTranscription.prompt.placeholder")}
              variant="compact"
            />
          </SettingContainer>

          <SettingContainer
            title={t("settings.remoteTranscription.timeout.title")}
            description={t("settings.remoteTranscription.timeout.description")}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <Input
              type="number"
              min={1}
              max={600}
              defaultValue={
                getSetting("remote_transcription_timeout_seconds") ?? 30
              }
              onBlur={(event) => {
                const seconds = Number.parseInt(event.target.value, 10);
                if (Number.isFinite(seconds) && seconds > 0) {
                  void updateSetting(
                    "remote_transcription_timeout_seconds",
                    seconds,
                  );
                }
              }}
              variant="compact"
              className="w-24"
            />
          </SettingContainer>

          <ToggleSwitch
            checked={
              getSetting("remote_transcription_fallback_to_local") ?? true
            }
            onChange={(value) =>
              void updateSetting(
                "remote_transcription_fallback_to_local",
                value,
              )
            }
            isUpdating={isUpdating("remote_transcription_fallback_to_local")}
            label={t("settings.remoteTranscription.fallback.label")}
            description={t("settings.remoteTranscription.fallback.description")}
            descriptionMode="tooltip"
            grouped={true}
          />

          <SettingContainer
            title={t("settings.remoteTranscription.test.title")}
            description={t("settings.remoteTranscription.test.description")}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <div className="flex items-center gap-2">
              {testState.status === "ok" && (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              )}
              <Button
                onClick={handleTest}
                variant="secondary"
                size="md"
                disabled={isTesting}
              >
                {isTesting
                  ? t("settings.remoteTranscription.test.running")
                  : t("settings.remoteTranscription.test.action")}
              </Button>
            </div>
          </SettingContainer>

          {testState.status === "error" && (
            <Alert variant="error" contained>
              {testState.message}
            </Alert>
          )}
        </>
      )}
    </SettingsGroup>
  );
};

export const RemoteTranscriptionSettings = React.memo(
  RemoteTranscriptionSettingsComponent,
);
RemoteTranscriptionSettings.displayName = "RemoteTranscriptionSettings";
