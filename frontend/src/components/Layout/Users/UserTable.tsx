// 파일: ~/HealthcareAI/frontend/src/components/Users/UserTable.tsx
import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getPaginationRowModel,
  getSortedRowModel,
  getFilteredRowModel,
} from '@tanstack/react-table';
import { usersAPI } from '@/lib/api';
import type { User } from '@/types';
import { format } from 'date-fns';
import toast from 'react-hot-toast';

const columnHelper = createColumnHelper<User>();

export const UserTable: React.FC = () => {
  const [globalFilter, setGlobalFilter] = useState('');
  const queryClient = useQueryClient();

  const { data: users = [], isLoading } = useQuery({
    queryKey: ['users'],
    queryFn: () => usersAPI.getUsers(0, 1000),
  });

  const deleteUserMutation = useMutation({
    mutationFn: usersAPI.deleteUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      toast.success('User deleted successfully');
    },
    onError: () => {
      toast.error('Failed to delete user');
    },
  });

  const columns = [
    columnHelper.accessor('full_name', {
      header: 'Name',
      cell: (info) => (
        <div className="flex items-center">
          <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center mr-3">
            <span className="text-sm font-medium text-primary-600">
              {info.getValue()?.charAt(0).toUpperCase()}
            </span>
          </div>
          <div>
            <div className="font-medium text-gray-900">{info.getValue()}</div>
            <div className="text-sm text-gray-500">{info.row.original.email}</div>
          </div>
        </div>
      ),
    }),
    columnHelper.accessor('is_active', {
      header: 'Status',
      cell: (info) => (
        <span
          className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
            info.getValue()
              ? 'bg-green-100 text-green-800'
              : 'bg-red-100 text-red-800'
          }`}
        >
          {info.getValue() ? 'Active' : 'Inactive'}
        </span>
      ),
    }),
    columnHelper.accessor('is_superuser', {
      header: 'Role',
      cell: (info) => (
        <span className="text-sm text-gray-900">
          {info.getValue() ? 'Admin' : 'User'}
        </span>
      ),
    }),
    columnHelper.accessor('created_at', {
      header: 'Joined',
      cell: (info) => (
        <span className="text-sm text-gray-500">
          {format(new Date(info.getValue()), 'MMM dd, yyyy')}
        </span>
      ),
    }),
    columnHelper.display({
      id: 'actions',
      header: 'Actions',
      cell: (info) => (
        <div className="flex space-x-2">
          <button
            onClick={() => handleViewUser(info.row.original.id)}
            className="text-primary-600 hover:text-primary-800 text-sm font-medium"
          >
            View
          </button>
          <button
            onClick={() => handleEditUser(info.row.original.id)}
            className="text-gray-600 hover:text-gray-800 text-sm font-medium"
          >
            Edit
          </button>
          <button
            onClick={() => handleDeleteUser(info.row.original.id)}
            className="text-red-600 hover:text-red-800 text-sm font-medium"
          >
            Delete
          </button>
        </div>
      ),
    }),
  ];

  const table = useReactTable({
    data: users,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: 'includesString',
    state: {
      globalFilter,
    },
    onGlobalFilterChange: setGlobalFilter,
  });

  const handleViewUser = (userId: string) => {
    // Navigate to user detail page
    window.location.href = `/admin/users/${userId}`;
  };

  const handleEditUser = (userId: string) => {
    // Open edit modal or navigate to edit page
    console.log('Edit user:', userId);
  };

  const handleDeleteUser = (userId: string) => {
    if (window.confirm('Are you sure you want to delete this user?')) {
      deleteUserMutation.mutate(userId);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="flex justify-between items-center">
        <input
          value={globalFilter ?? ''}
          onChange={(e) => setGlobalFilter(String(e.target.value))}
          className="form-input max-w-sm"
          placeholder="Search users..."
        />
        <button className="btn btn-primary">
          Add User
        </button>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="hover:bg-gray-50">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-6 py-4 whitespace-nowrap">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-700">
          Showing {table.getState().pagination.pageIndex * table.getState().pagination.pageSize + 1} to{' '}
          {Math.min(
            (table.getState().pagination.pageIndex + 1) * table.getState().pagination.pageSize,
            table.getFilteredRowModel().rows.length
          )}{' '}
          of {table.getFilteredRowModel().rows.length} results
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
            className="btn btn-secondary disabled:opacity-50"
          >
            Previous
          </button>
          <button
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
            className="btn btn-secondary disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
};
