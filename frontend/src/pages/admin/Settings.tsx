// 파일: ~/HealthcareAI/frontend/src/pages/admin/Settings.tsx
import React, { useState } from 'react';
import toast from 'react-hot-toast';

export const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState({
    notifications: {
      email: true,
      push: true,
      workout_reminders: true,
      challenge_updates: true,
    },
    security: {
      two_factor: false,
      session_timeout: '24',
      password_policy: 'medium',
    },
    ai: {
      pose_sensitivity: '0.8',
      feedback_level: 'detailed',
      auto_correction: true,
    },
    system: {
      maintenance_mode: false,
      debug_mode: false,
      analytics: true,
    },
  });

  const handleSave = () => {
    // API call to save settings would go here
    toast.success('Settings saved successfully!');
  };

  const handleSettingChange = (section: string, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section as keyof typeof prev],
        [key]: value,
      },
    }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600">Configure platform settings and preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Notifications */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Notifications</h3>
          <div className="space-y-4">
            {Object.entries(settings.notifications).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 capitalize">
                  {key.replace('_', ' ')}
                </label>
                <button
                  onClick={() => handleSettingChange('notifications', key, !value)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    value ? 'bg-primary-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      value ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Security */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Security</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">
                Two-Factor Authentication
              </label>
              <button
                onClick={() => handleSettingChange('security', 'two_factor', !settings.security.two_factor)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  settings.security.two_factor ? 'bg-primary-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    settings.security.two_factor ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Session Timeout (hours)
              </label>
              <select
                value={settings.security.session_timeout}
                onChange={(e) => handleSettingChange('security', 'session_timeout', e.target.value)}
                className="form-input"
              >
                <option value="1">1 hour</option>
                <option value="8">8 hours</option>
                <option value="24">24 hours</option>
                <option value="168">1 week</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Password Policy
              </label>
              <select
                value={settings.security.password_policy}
                onChange={(e) => handleSettingChange('security', 'password_policy', e.target.value)}
                className="form-input"
              >
                <option value="low">Low (6+ characters)</option>
                <option value="medium">Medium (8+ with symbols)</option>
                <option value="high">High (12+ complex)</option>
              </select>
            </div>
          </div>
        </div>

        {/* AI Settings */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Configuration</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Pose Detection Sensitivity
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={settings.ai.pose_sensitivity}
                onChange={(e) => handleSettingChange('ai', 'pose_sensitivity', e.target.value)}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Low</span>
                <span>{settings.ai.pose_sensitivity}</span>
                <span>High</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Feedback Level
              </label>
              <select
                value={settings.ai.feedback_level}
                onChange={(e) => handleSettingChange('ai', 'feedback_level', e.target.value)}
                className="form-input"
              >
                <option value="minimal">Minimal</option>
                <option value="standard">Standard</option>
                <option value="detailed">Detailed</option>
              </select>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">
                Auto Pose Correction
              </label>
              <button
                onClick={() => handleSettingChange('ai', 'auto_correction', !settings.ai.auto_correction)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  settings.ai.auto_correction ? 'bg-primary-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    settings.ai.auto_correction ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>

        {/* System */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System</h3>
          <div className="space-y-4">
            {Object.entries(settings.system).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 capitalize">
                  {key.replace('_', ' ')}
                </label>
                <button
                  onClick={() => handleSettingChange('system', key, !value)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    value ? 'bg-primary-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      value ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <button onClick={handleSave} className="btn btn-primary">
          Save Settings
        </button>
      </div>
    </div>
  );
};