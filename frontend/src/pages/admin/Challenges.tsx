// 파일: ~/HealthcareAI/frontend/src/pages/admin/Challenges.tsx
import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { challengesAPI } from '@/lib/api';
import { Challenge } from '@/types';
import { format } from 'date-fns';
import toast from 'react-hot-toast';

export const ChallengesPage: React.FC = () => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const queryClient = useQueryClient();

  const { data: challenges = [], isLoading } = useQuery({
    queryKey: ['challenges'],
    queryFn: challengesAPI.getChallenges,
  });

  const createMutation = useMutation({
    mutationFn: challengesAPI.createChallenge,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['challenges'] });
      setShowCreateModal(false);
      toast.success('Challenge created successfully!');
    },
    onError: () => {
      toast.error('Failed to create challenge');
    },
  });

  const handleCreateChallenge = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    
    const challenge = {
      title: formData.get('title') as string,
      description: formData.get('description') as string,
      target_value: parseInt(formData.get('target_value') as string),
      start_date: formData.get('start_date') as string,
      end_date: formData.get('end_date') as string,
      is_active: true,
    };

    createMutation.mutate(challenge);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Challenges</h1>
          <p className="text-gray-600">Manage fitness challenges and competitions</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn btn-primary"
        >
          Create Challenge
        </button>
      </div>

      {/* Challenges Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {challenges.map((challenge) => (
          <div key={challenge.id} className="card p-6">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-lg font-semibold text-gray-900">{challenge.title}</h3>
              <span className={`px-2 py-1 text-xs rounded-full ${
                challenge.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
              }`}>
                {challenge.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
            
            <p className="text-gray-600 text-sm mb-4">{challenge.description}</p>
            
            <div className="space-y-2 mb-4">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Progress</span>
                <span className="font-medium">
                  {challenge.current_value} / {challenge.target_value}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-primary-600 h-2 rounded-full"
                  style={{
                    width: `${Math.min((challenge.current_value / challenge.target_value) * 100, 100)}%`
                  }}
                ></div>
              </div>
            </div>
            
            <div className="space-y-1 text-sm text-gray-500">
              <div>Participants: {challenge.participants_count}</div>
              <div>Ends: {format(new Date(challenge.end_date), 'MMM dd, yyyy')}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Create Challenge Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">Create New Challenge</h2>
            
            <form onSubmit={handleCreateChallenge} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Title
                </label>
                <input
                  name="title"
                  type="text"
                  required
                  className="form-input"
                  placeholder="30-Day Push-up Challenge"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  name="description"
                  required
                  rows={3}
                  className="form-input"
                  placeholder="Complete 1000 push-ups in 30 days"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Target Value
                </label>
                <input
                  name="target_value"
                  type="number"
                  required
                  className="form-input"
                  placeholder="1000"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Start Date
                  </label>
                  <input
                    name="start_date"
                    type="date"
                    required
                    className="form-input"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    End Date
                  </label>
                  <input
                    name="end_date"
                    type="date"
                    required
                    className="form-input"
                  />
                </div>
              </div>
              
              <div className="flex space-x-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="btn btn-secondary flex-1"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={createMutation.isPending}
                  className="btn btn-primary flex-1"
                >
                  {createMutation.isPending ? 'Creating...' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};