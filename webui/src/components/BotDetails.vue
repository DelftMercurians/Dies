<script setup lang="ts"></script>

<template>
  <v-card width="70%" height="40%">
    <template v-slot:title> Details </template>

    <template v-slot:subtitle>
      <p>Id:</p>
      <p>Position:</p>
      <p>Health:</p>
    </template>

    <template v-slot:text> This is content </template>

    <v-card-actions>
      <v-switch label="Trajectory" color="green"></v-switch>
    </v-card-actions>
  </v-card>
</template>

<style scoped></style>
